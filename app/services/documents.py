from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.vector_store import QdrantVectorStore


class DocumentNotFoundError(LookupError):
    pass


class DocumentService:
    """Read and delete ingested documents, keeping SQL metadata and Qdrant vectors in sync."""

    def __init__(self, db: Session, vector_store: QdrantVectorStore) -> None:
        self._db = db
        self._vector_store = vector_store

    def list_all(self, session_id: str | None = None) -> list[Document]:
        query = select(Document).order_by(Document.created_at.desc())
        if session_id is not None:
            query = query.where(Document.session_id == session_id)
        return list(self._db.scalars(query))

    def ids_for_session(self, session_id: str) -> list[str]:
        return list(self._db.scalars(select(Document.id).where(Document.session_id == session_id)))

    def delete(self, document_id: str) -> None:
        document = self._db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        # Remove vectors first: if this fails, the metadata row remains and the delete can be retried.
        self._vector_store.delete_document(document_id)
        self._db.delete(document)
        self._db.commit()
