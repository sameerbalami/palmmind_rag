import logging
import time
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

EmbeddingTask = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_EMBED_BATCH_SIZE = 100  # Gemini's per-request limit for embed_content
_RETRYABLE_STATUS = frozenset({429, 500, 503})
_DEFAULT_RETRY_DELAY_SECONDS = 60.0


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(f"LLM provider rate limit reached, retry in {retry_after_seconds:.0f}s")
        self.retry_after_seconds = retry_after_seconds


class GeminiClient:
    """Thin wrapper over the Gemini SDK: embeddings, free-text generation and structured (JSON) generation."""

    def __init__(
        self,
        api_key: str,
        chat_models: list[str],
        embedding_model: str,
        embedding_dim: int,
        timeout_seconds: int = 60,
        max_rate_limit_wait_seconds: int = 120,
    ) -> None:
        if not chat_models:
            raise ValueError("At least one chat model is required")
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )
        self._chat_models = chat_models
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._max_rate_limit_wait_seconds = max_rate_limit_wait_seconds

    def embed(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self._embedding_dim)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            response = self._embed_batch(texts[start : start + _EMBED_BATCH_SIZE], config)
            if not response.embeddings:
                raise LLMError("Embedding response was empty")
            vectors.extend(list(e.values or []) for e in response.embeddings)
        return vectors

    def _embed_batch(self, batch: list[str], config: types.EmbedContentConfig) -> types.EmbedContentResponse:
        """Embed one batch, sleeping through per-minute rate limits (HTTP 429) within a total wait budget.

        The free tier counts every text in a batch as one request, so large documents hit the limit quickly.
        """
        waited = 0.0
        while True:
            try:
                return self._client.models.embed_content(model=self._embedding_model, contents=batch, config=config)
            except genai_errors.APIError as exc:
                if exc.code != 429:
                    raise
                delay = _retry_delay_seconds(exc)
                if waited + delay > self._max_rate_limit_wait_seconds:
                    raise RateLimitError(retry_after_seconds=delay) from exc
                logger.warning("Embedding rate limit hit, waiting %.0fs before retrying", delay)
                time.sleep(delay)
                waited += delay

    def generate(self, system_prompt: str, messages: list[ChatMessage], temperature: float = 0.3) -> str:
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature)
        return self._generate_text(messages, config).strip()

    def generate_structured[T: BaseModel](self, system_prompt: str, messages: list[ChatMessage], schema: type[T]) -> T:
        """Ask the model for JSON matching `schema` and parse it into that Pydantic model."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        return schema.model_validate_json(self._generate_text(messages, config))

    def _generate_text(self, messages: list[ChatMessage], config: types.GenerateContentConfig) -> str:
        """Call the primary chat model, falling back to the next one while models are overloaded."""
        contents = _to_contents(messages)
        last_error: genai_errors.APIError | None = None
        for model in self._chat_models:
            try:
                response = self._client.models.generate_content(model=model, contents=contents, config=config)
            except genai_errors.APIError as exc:
                if exc.code not in _RETRYABLE_STATUS:
                    raise
                logger.warning("Chat model %s unavailable (%s), trying next fallback", model, exc.code)
                last_error = exc
                continue
            if not response.text:
                raise LLMError(f"{model} returned an empty response")
            return response.text
        raise LLMError(f"All chat models are unavailable: {last_error}")


def _retry_delay_seconds(exc: genai_errors.APIError) -> float:
    """Read the server-suggested wait (RetryInfo.retryDelay, e.g. "49s") from a 429 error."""
    details = exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
    for detail in details:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            try:
                return float(str(detail.get("retryDelay", "")).removesuffix("s")) + 1
            except ValueError:
                break
    return _DEFAULT_RETRY_DELAY_SECONDS


def _to_contents(messages: list[ChatMessage]) -> list[types.Content]:
    # Gemini names the assistant role "model".
    return [
        types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
        for m in messages
    ]
