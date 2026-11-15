import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class QdrantVectorStore:
    """Stores chunk embeddings in Qdrant and runs cosine-similarity search over them."""

    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
        # Index the field used to filter searches by document.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def upsert_chunks(self, document_id: str, filename: str, chunks: list[str], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"document_id": document_id, "filename": filename, "chunk_index": i, "text": chunk},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=_document_filter([document_id])),
            wait=True,
        )

    def search(self, vector: list[float], top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        query_filter = _document_filter(document_ids) if document_ids else None
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                document_id=str(p.payload["document_id"]),
                filename=str(p.payload["filename"]),
                chunk_index=int(p.payload["chunk_index"]),
                text=str(p.payload["text"]),
                score=p.score,
            )
            for p in result.points
            if p.payload is not None
        ]


def _document_filter(document_ids: list[str]) -> Filter:
    return Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))])
