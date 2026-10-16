"""Load PDF / Markdown / TXT files into plain text documents."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class Document:
    source: str  # file name, used in citations
    text: str


def load_file(path: str | Path) -> Document:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
    return Document(source=p.name, text=text)


def load_dir(folder: str | Path) -> list[Document]:
    files = sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in SUPPORTED)
    return [load_file(f) for f in files]
