from rag_eval.chunking import Chunk
from rag_eval.retrieval import BM25Retriever


def test_bm25_ranks_relevant_chunk_first():
    chunks = [Chunk("1", "a.md", "Leave", "Employees get 24 days of annual leave."),
              Chunk("2", "b.md", "Pricing", "The Team plan costs 4,500 INR.")]
    hits = BM25Retriever(chunks).search("how much annual leave", k=2)
    assert hits[0].chunk.id == "1"
    assert 0 < hits[0].score <= 1


def test_no_match_returns_empty():
    r = BM25Retriever([Chunk("1", "a.md", "", "apples and oranges")])
    assert r.search("quantum chromodynamics") == []
