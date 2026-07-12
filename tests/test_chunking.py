from src.rag.chunking import chunk_text


def test_chunking_returns_multiple_chunks() -> None:
    chunks = chunk_text("a" * 1000, chunk_size=300, overlap=50)
    assert len(chunks) >= 4
    assert all(chunks)
