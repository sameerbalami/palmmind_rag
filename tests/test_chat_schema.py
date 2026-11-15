import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest


def test_accepts_uuid_session_id() -> None:
    request = ChatRequest(session_id="0c99357f-7bfd-4fe6-9648-b9a4a129dce2", message="hi")
    assert request.session_id == "0c99357f-7bfd-4fe6-9648-b9a4a129dce2"


def test_session_id_is_optional() -> None:
    assert ChatRequest(message="hi").session_id is None


@pytest.mark.parametrize("bad", ["0c99357f-7bfd-4fe6-9648-b9a4a129dce2>", "<paste session_id>", "a b", ""])
def test_rejects_malformed_session_id(bad: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id=bad, message="hi")
