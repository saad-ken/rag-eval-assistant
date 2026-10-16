"""The RAG pipeline: load -> chunk -> index -> retrieve -> prompt -> answer with citations."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from .chunking import chunk_documents
from .config import Settings
from .loaders import Document, load_dir
from .prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from .providers import get_llm
from .retrieval import Hit, make_retriever


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    refused: bool
    citations: list[str] = field(default_factory=list)  # chunk labels actually cited

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text, "refused": self.refused,
                "citations": self.citations,
                "sources": [{"label": h.chunk.label, "score": round(h.score, 3), "text": h.chunk.text}
                            for h in self.hits]}


def is_refusal(text: str) -> bool:
    t = text.lower()
    return "i don't know" in t or "i do not know" in t or "not covered" in t


class RAGAssistant:
    def __init__(self, docs: list[Document], settings: Settings | None = None, llm=None):
        self.settings = settings or Settings.from_env()
        self.chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        self.retriever = make_retriever(self.chunks, self.settings.embeddings)
        self.llm = llm or get_llm(self.settings.llm_provider, self.settings.llm_model)

    @classmethod
    def from_folder(cls, folder: str | Path, settings: Settings | None = None, llm=None):
        docs = load_dir(folder)
        if not docs:
            raise ValueError(f"No PDF/MD/TXT files found in {folder}")
        return cls(docs, settings, llm)

    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.settings.top_k)
        # Guardrail 1: weak retrieval => refuse before the LLM can guess.
        if not hits or hits[0].score < self.settings.min_score:
            return Answer(question, REFUSAL, hits, True)
        contexts = [f"({h.chunk.label}) {h.chunk.text}" for h in hits]
        text = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, contexts), contexts, question)
        refused = is_refusal(text)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        return Answer(question, text, hits, refused, [hits[n - 1].chunk.label for n in cited])
