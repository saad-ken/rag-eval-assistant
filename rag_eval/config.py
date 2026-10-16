"""Settings read from environment variables (see .env.example)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Settings:
    llm_provider: str = "mock"
    llm_model: str = ""
    embeddings: str = "bm25"
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 100
    min_score: float = 0.15  # below this retrieval score we refuse instead of guessing

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        e = os.environ.get
        return cls(
            llm_provider=e("LLM_PROVIDER", "mock").lower(),
            llm_model=e("LLM_MODEL", ""),
            embeddings=e("EMBEDDINGS", "bm25").lower(),
            top_k=int(e("TOP_K", "4")),
            chunk_size=int(e("CHUNK_SIZE", "600")),
            chunk_overlap=int(e("CHUNK_OVERLAP", "100")),
            min_score=float(e("MIN_SCORE", "0.15")),
        )
