from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.booking import BookingOut

# Letters, digits, "-" and "_" only, so a stray character from copy-paste fails loudly
# instead of silently starting a new conversation.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "How long is the internship?"},
                {"session_id": "<session_id from a previous response>", "message": "Is it paid?"},
                {"message": "What are the candidate's skills?", "document_ids": ["<id from POST /documents>"]},
            ]
        }
    )

    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=SESSION_ID_PATTERN,
        description="Conversation id. Omit to start a new conversation; reuse the returned id for follow-ups.",
    )
    message: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Only retrieve from these documents. Omit to search every uploaded document.",
    )


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


Intent = Literal["question", "booking", "booking_status"]


class ChatResponse(BaseModel):
    session_id: str
    intent: Intent
    answer: str
    sources: list[SourceChunk] = []
    booking: BookingOut | None = Field(default=None, description="The booking created by this message, if any")
    bookings: list[BookingOut] = Field(default=[], description="Upcoming bookings, for booking_status questions")
