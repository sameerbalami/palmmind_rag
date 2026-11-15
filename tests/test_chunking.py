import pytest

from app.schemas.document import ChunkingStrategy
from app.services.chunking import chunk_text, fixed_size_chunks, sentence_chunks

TEXT = "Palm Mind builds AI products. It is based in Nepal. The team works on RAG systems! Do they hire interns?"


def test_fixed_size_chunks_respects_size_and_overlap() -> None:
    chunks = fixed_size_chunks("abcdefghij", chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_fixed_size_chunks_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks("abc", chunk_size=4, overlap=4)


def test_sentence_chunks_keep_sentences_whole() -> None:
    chunks = sentence_chunks(TEXT, max_chars=60)
    assert chunks == [
        "Palm Mind builds AI products. It is based in Nepal.",
        "The team works on RAG systems! Do they hire interns?",
    ]


def test_sentence_chunks_split_oversized_sentence() -> None:
    chunks = sentence_chunks("x" * 25, max_chars=10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]
    assert all(len(c) <= 10 for c in chunks)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunk_text_dispatches_every_strategy(strategy: ChunkingStrategy) -> None:
    chunks = chunk_text(TEXT, strategy, chunk_size=50, overlap=10)
    assert chunks
    assert all(len(c) <= 50 for c in chunks)


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   ", ChunkingStrategy.SENTENCE, chunk_size=50, overlap=10) == []
    assert chunk_text("   ", ChunkingStrategy.FIXED, chunk_size=50, overlap=10) == []
