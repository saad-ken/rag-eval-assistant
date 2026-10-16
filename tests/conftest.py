import os, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDINGS"] = "bm25"
from rag_eval.config import Settings
from rag_eval.pipeline import RAGAssistant


@pytest.fixture(scope="session")
def bot():
    return RAGAssistant.from_folder(ROOT / "data/sample_docs", Settings())
