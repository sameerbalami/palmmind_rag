import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.document import ChunkingStrategy
from app.services.chunking import chunk_text
from app.services.llm import GeminiClient
from app.services.text_extraction import SUPPORTED_EXTENSIONS, UnsupportedFileError, extract_text
from app.services.vector_store import QdrantVectorStore


class EmptyDocumentError(ValueError):
    pass


class IngestionService:
    """Upload pipeline: extract text -> chunk -> embed -> store vectors (Qdrant) -> store metadata (SQL)."""

    def __init__(
        self,
        db: Session,
        llm: GeminiClient,
        vector_store: QdrantVectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._db = db
        self._llm = llm
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: ChunkingStrategy,
        session_id: str,
    ) -> Document:
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"Only {sorted(SUPPORTED_EXTENSIONS)} files are supported")

        text = extract_text(filename, data)
        chunks = chunk_text(text, strategy, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise EmptyDocumentError("No extractable text found in the file")

        document_id = str(uuid.uuid4())
        vectors = self._llm.embed(chunks, task="RETRIEVAL_DOCUMENT")
        self._vector_store.upsert_chunks(document_id, filename, chunks, vectors)

        document = Document(
            id=document_id,
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            chunking_strategy=strategy.value,
            chunk_count=len(chunks),
            char_count=len(text),
        )
        self._db.add(document)
        self._db.commit()
        return document
