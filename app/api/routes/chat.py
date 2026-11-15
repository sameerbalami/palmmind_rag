from fastapi import APIRouter, status

from app.api.deps import ChatServiceDep, MemoryDep
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Send a message. Answers questions from uploaded documents, or books an interview when asked."""
    return service.chat(request.message, request.session_id, request.document_ids)


@router.get("/{session_id}/history", response_model=list[ChatMessage])
def get_history(session_id: str, memory: MemoryDep) -> list[ChatMessage]:
    return memory.get_history(session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(session_id: str, memory: MemoryDep) -> None:
    memory.clear(session_id)
