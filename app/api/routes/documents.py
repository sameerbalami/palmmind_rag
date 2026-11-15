import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.api.deps import DocumentServiceDep, IngestionServiceDep, SettingsDep
from app.models.document import Document
from app.schemas.chat import SESSION_ID_PATTERN
from app.schemas.document import ChunkingStrategy, DocumentOut
from app.services.documents import DocumentNotFoundError
from app.services.ingestion import EmptyDocumentError
from app.services.text_extraction import UnsupportedFileError

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    service: IngestionServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.SENTENCE,
    session_id: Annotated[
        str | None,
        Form(
            max_length=64,
            pattern=SESSION_ID_PATTERN,
            description="Chat session to attach the document to. Omit to start a new session. "
            "Chat in that session answers only from its documents.",
        ),
    ] = None,
) -> Document:
    """Upload a document, chunk it with the chosen strategy, embed the chunks and index them."""
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File must have a name")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB")

    try:
        return service.ingest(
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            data=data,
            strategy=chunking_strategy,
            session_id=session_id or str(uuid.uuid4()),
        )
    except UnsupportedFileError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    service: DocumentServiceDep,
    session_id: Annotated[str | None, Query(description="Only documents attached to this session")] = None,
) -> list[Document]:
    return service.list_all(session_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, service: DocumentServiceDep) -> None:
    """Delete a document's metadata and all of its vectors."""
    try:
        service.delete(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
