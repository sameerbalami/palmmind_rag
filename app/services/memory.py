from redis import Redis

from app.schemas.chat import ChatMessage


class RedisChatMemory:
    """Per-session chat history stored as a Redis list of JSON messages.

    The list is capped at `max_messages` and expires after `ttl_seconds` of inactivity.
    """

    def __init__(self, redis: Redis, max_messages: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    def get_history(self, session_id: str) -> list[ChatMessage]:
        raw: list[str] = self._redis.lrange(self._key(session_id), 0, -1)  # type: ignore[assignment]
        return [ChatMessage.model_validate_json(item) for item in raw]

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, *(m.model_dump_json() for m in messages))
        pipe.ltrim(key, -self._max_messages, -1)
        pipe.expire(key, self._ttl_seconds)
        pipe.execute()

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))
