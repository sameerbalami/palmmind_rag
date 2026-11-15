import uuid
from datetime import date, datetime, time

from pydantic import BaseModel, Field, ValidationError

from app.models.booking import Booking
from app.schemas.booking import BookingCreate, BookingOut
from app.schemas.chat import ChatMessage, ChatResponse, Intent, SourceChunk
from app.services.booking import BookingConflictError, BookingService
from app.services.documents import DocumentService
from app.services.llm import GeminiClient
from app.services.memory import RedisChatMemory
from app.services.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    BOOKING_STATUS_SYSTEM_PROMPT,
    NO_BOOKINGS,
    NO_CONTEXT,
    RAG_SYSTEM_PROMPT,
)
from app.services.vector_store import QdrantVectorStore, RetrievedChunk


class ConversationAnalysis(BaseModel):
    """Structured output of the first LLM call on every turn."""

    intent: Intent
    standalone_question: str
    name: str | None = Field(default=None, description="Full name of the person booking")
    email: str | None = Field(default=None, description="Email address")
    date: str | None = Field(default=None, description="Interview date, YYYY-MM-DD")
    time: str | None = Field(default=None, description="Interview time, 24-hour HH:MM")


class ChatService:
    """Custom conversational RAG with Redis memory and LLM-driven interview booking.

    Each turn:
      1. Load the session history from Redis.
      2. One structured LLM call classifies intent, rewrites the message into a standalone question
         and extracts booking fields.
      3a. booking        -> ask for missing fields, or validate and store the booking.
      3b. booking_status -> look up upcoming bookings (this session, or the email the user gives) and answer
                            from those facts; days remaining are computed in code, not by the LLM.
      3c. question       -> embed the standalone question, retrieve top-k chunks from Qdrant,
                      and generate an answer grounded in those chunks plus the history.
                      Retrieval scope: explicit `document_ids`, else the documents uploaded in this
                      session, else every document.
      4. Append the user message and the answer to Redis.
    """

    def __init__(
        self,
        llm: GeminiClient,
        vector_store: QdrantVectorStore,
        memory: RedisChatMemory,
        bookings: BookingService,
        documents: DocumentService,
        top_k: int,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._memory = memory
        self._bookings = bookings
        self._documents = documents
        self._top_k = top_k

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResponse:
        session_id = session_id or str(uuid.uuid4())
        history = self._memory.get_history(session_id)
        user_message = ChatMessage(role="user", content=message)

        analysis = self._analyze(history, user_message)
        match analysis.intent:
            case "booking":
                response = self._handle_booking(session_id, analysis)
            case "booking_status":
                response = self._answer_booking_status(session_id, history, user_message, analysis.email)
            case "question":
                response = self._answer_question(
                    session_id, history, user_message, analysis.standalone_question, document_ids
                )

        self._memory.append(session_id, user_message, ChatMessage(role="assistant", content=response.answer))
        return response

    def _analyze(self, history: list[ChatMessage], user_message: ChatMessage) -> ConversationAnalysis:
        today = date.today()
        system_prompt = ANALYZE_SYSTEM_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"))
        return self._llm.generate_structured(system_prompt, [*history, user_message], ConversationAnalysis)

    def _answer_question(
        self,
        session_id: str,
        history: list[ChatMessage],
        user_message: ChatMessage,
        standalone_question: str,
        document_ids: list[str] | None,
    ) -> ChatResponse:
        scope = document_ids or self._documents.ids_for_session(session_id) or None
        query_vector = self._llm.embed([standalone_question], task="RETRIEVAL_QUERY")[0]
        chunks = self._vector_store.search(query_vector, self._top_k, scope)

        system_prompt = RAG_SYSTEM_PROMPT.format(context=_format_context(chunks))
        answer = self._llm.generate(system_prompt, [*history, user_message])

        return ChatResponse(
            session_id=session_id,
            intent="question",
            answer=answer,
            sources=[
                SourceChunk(
                    document_id=c.document_id,
                    filename=c.filename,
                    chunk_index=c.chunk_index,
                    score=c.score,
                    text=c.text,
                )
                for c in chunks
            ],
        )

    def _answer_booking_status(
        self,
        session_id: str,
        history: list[ChatMessage],
        user_message: ChatMessage,
        email: str | None,
    ) -> ChatResponse:
        bookings = self._bookings.find_upcoming(session_id, email)
        now = datetime.now()
        facts = "\n".join(f"- {describe_booking(b, now.date())}" for b in bookings) or NO_BOOKINGS
        system_prompt = BOOKING_STATUS_SYSTEM_PROMPT.format(now=f"{now:%A, %d %B %Y, %I:%M %p}", bookings=facts)
        answer = self._llm.generate(system_prompt, [*history, user_message])
        return ChatResponse(
            session_id=session_id,
            intent="booking_status",
            answer=answer,
            bookings=[BookingOut.model_validate(b) for b in bookings],
        )

    def _handle_booking(self, session_id: str, analysis: ConversationAnalysis) -> ChatResponse:
        fields = {"name": analysis.name, "email": analysis.email, "date": analysis.date, "time": analysis.time}
        missing = [field for field, value in fields.items() if not value]
        if missing:
            answer = f"I'd be happy to book your interview. Please provide your {_human_join(missing)}."
            return ChatResponse(session_id=session_id, intent="booking", answer=answer)

        try:
            booking_data = BookingCreate(session_id=session_id, **fields)  # type: ignore[arg-type]
            booking = self._bookings.create(booking_data)
        except ValidationError as exc:
            problems = "; ".join(_clean_error(e["msg"]) for e in exc.errors())
            answer = f"I couldn't book that yet: {problems}. Please correct it and try again."
            return ChatResponse(session_id=session_id, intent="booking", answer=answer)
        except BookingConflictError as exc:
            return ChatResponse(session_id=session_id, intent="booking", answer=f"{exc}.")

        answer = (
            f"Your interview is booked, {booking.name}! "
            f"Date: {booking.date:%A, %d %B %Y} at {_format_time(booking.time)}. "
            f"Booking reference: {booking.id[:8]} (contact email: {booking.email})."
        )
        return ChatResponse(
            session_id=session_id,
            intent="booking",
            answer=answer,
            booking=BookingOut.model_validate(booking),
        )


def describe_booking(booking: Booking, today: date) -> str:
    """One-line fact about a booking, with the days remaining computed here rather than by the LLM."""
    days = (booking.date - today).days
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return (
        f"{booking.date:%A, %d %B %Y} at {_format_time(booking.time)} ({when}); "
        f"name: {booking.name}; email: {booking.email}; reference: {booking.id[:8]}"
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return NO_CONTEXT
    return "\n\n".join(f"[{i}] (source: {c.filename})\n{c.text}" for i, c in enumerate(chunks, start=1))


def _human_join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _format_time(value: time) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


def _clean_error(message: str) -> str:
    return message.removeprefix("Value error, ")
