import re

from app.schemas.document import ChunkingStrategy

_WHITESPACE = re.compile(r"[ \t]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into windows of `chunk_size` characters, each overlapping the previous by `overlap`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = normalize(text)
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def sentence_chunks(text: str, max_chars: int) -> list[str]:
    """Group whole sentences into chunks of at most `max_chars` characters.

    A single sentence longer than `max_chars` is split with `fixed_size_chunks` so no chunk exceeds the limit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    sentences = [normalize(s) for s in _SENTENCE_BOUNDARY.split(text)]
    chunks: list[str] = []
    current = ""
    for sentence in filter(None, sentences):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_size_chunks(sentence, max_chars, overlap=0))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> list[str]:
    match strategy:
        case ChunkingStrategy.FIXED:
            return fixed_size_chunks(text, chunk_size, overlap)
        case ChunkingStrategy.SENTENCE:
            return sentence_chunks(text, chunk_size)
