from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    """Fixed-size character windows with overlap."""

    SENTENCE = "sentence"
    """Whole sentences packed into chunks up to a size limit."""


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str | None
    filename: str
    content_type: str
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    char_count: int
    created_at: datetime
