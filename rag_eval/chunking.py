"""Split documents into overlapping chunks, preferring paragraph boundaries.

Each chunk remembers its source file and nearest Markdown heading so
citations can say *where* an answer came from, e.g. handbook.md > Leave Policy.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from .loaders import Document


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} > {self.section}" if self.section else self.source


def chunk_document(doc: Document, size: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    section, buf = "", ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(Chunk(f"{doc.source}#{len(chunks)}", doc.source, section, buf.strip()))
        buf = buf[-overlap:] if overlap and len(buf) > overlap else ""

    for para in re.split(r"\n\s*\n", doc.text):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", para)
        if m:  # new heading: close current chunk so chunks don't straddle sections
            if buf.strip():
                flush()
            buf = ""
            section = m.group(1).split("\n")[0].strip()
            para = para[m.end():].strip()
            if not para:
                continue
        while len(para) > size:  # very long paragraph: hard split
            buf += para[:size]
            para = para[size - overlap:]
            flush()
        if len(buf) + len(para) > size:
            flush()
        buf = (buf + "\n\n" + para) if buf else para
    if buf.strip():
        chunks.append(Chunk(f"{doc.source}#{len(chunks)}", doc.source, section, buf.strip()))
    return chunks


def chunk_documents(docs: list[Document], size: int = 600, overlap: int = 100) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, size, overlap))
    return out
