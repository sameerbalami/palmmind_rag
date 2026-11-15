from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

from app.services import llm as llm_module
from app.services.llm import GeminiClient, RateLimitError


def _rate_limit_error(retry_delay: str = "49s") -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}],
            }
        },
    )


class _FakeModels:
    """Fails with 429 for the first `failures` calls, then returns one vector per input text."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> SimpleNamespace:
        self.calls += 1
        if self.calls <= self.failures:
            raise _rate_limit_error()
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents])


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", recorded.append)
    return recorded


def _client(models: _FakeModels, max_wait: int) -> GeminiClient:
    client = GeminiClient("test-key", ["chat-model"], "embed-model", 2, max_rate_limit_wait_seconds=max_wait)
    client._client = SimpleNamespace(models=models)  # type: ignore[assignment]
    return client


def test_embed_waits_for_rate_limit_then_succeeds(sleeps: list[float]) -> None:
    models = _FakeModels(failures=1)
    vectors = _client(models, max_wait=120).embed(["a", "b"], task="RETRIEVAL_DOCUMENT")

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert models.calls == 2
    assert sleeps == [50.0]  # server-suggested 49s plus a 1s margin


def test_embed_raises_rate_limit_error_when_wait_budget_exceeded(sleeps: list[float]) -> None:
    models = _FakeModels(failures=10)
    with pytest.raises(RateLimitError) as exc_info:
        _client(models, max_wait=120).embed(["a"], task="RETRIEVAL_DOCUMENT")

    assert exc_info.value.retry_after_seconds == 50.0
    assert sleeps == [50.0, 50.0]  # a third 50s wait would exceed the 120s budget
