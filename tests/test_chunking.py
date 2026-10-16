from rag_eval.chunking import chunk_document
from rag_eval.loaders import Document, load_dir


def test_sections_are_tracked():
    doc = Document("x.md", "# A\n\nalpha text here\n\n## B\n\nbeta text here")
    chunks = chunk_document(doc, size=200, overlap=0)
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[1].label == "x.md > B"


def test_long_text_respects_size():
    doc = Document("y.txt", "word " * 1000)
    chunks = chunk_document(doc, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 360 for c in chunks)


def test_loads_all_sample_formats():
    names = {d.source for d in load_dir("data/sample_docs")}
    assert {"handbook.md", "product_faq.md", "release_test_plan.txt"} <= names
