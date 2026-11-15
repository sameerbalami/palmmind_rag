from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.ingestion import IngestionService
from app.services.llm import GeminiClient
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorStore

# Long-lived clients are created once per process and shared across requests.


@lru_cache
def get_llm() -> GeminiClient:
    s = get_settings()
    chat_models = list(dict.fromkeys([s.chat_model, *s.chat_fallback_models]))  # de-duplicated, order kept
    return GeminiClient(
        s.gemini_api_key,
        chat_models,
        s.embedding_model,
        s.embedding_dim,
        s.llm_timeout_seconds,
        s.llm_max_rate_limit_wait_seconds,
    )


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return QdrantVectorStore(client, s.qdrant_collection, s.embedding_dim)


@lru_cache
def get_memory() -> RedisChatMemory:
    s = get_settings()
    redis = Redis.from_url(s.redis_url, decode_responses=True)
    return RedisChatMemory(redis, s.chat_history_max_messages, s.chat_history_ttl_seconds)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]
LLMDep = Annotated[GeminiClient, Depends(get_llm)]
VectorStoreDep = Annotated[QdrantVectorStore, Depends(get_vector_store)]
MemoryDep = Annotated[RedisChatMemory, Depends(get_memory)]


def get_ingestion_service(
    db: DbDep, llm: LLMDep, vector_store: VectorStoreDep, settings: SettingsDep
) -> IngestionService:
    return IngestionService(db, llm, vector_store, settings.chunk_size, settings.chunk_overlap)


def get_booking_service(db: DbDep) -> BookingService:
    return BookingService(db)


def get_document_service(db: DbDep, vector_store: VectorStoreDep) -> DocumentService:
    return DocumentService(db, vector_store)


def get_chat_service(
    llm: LLMDep,
    vector_store: VectorStoreDep,
    memory: MemoryDep,
    bookings: Annotated[BookingService, Depends(get_booking_service)],
    documents: Annotated[DocumentService, Depends(get_document_service)],
    settings: SettingsDep,
) -> ChatService:
    return ChatService(llm, vector_store, memory, bookings, documents, settings.top_k)


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
BookingServiceDep = Annotated[BookingService, Depends(get_booking_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
