"""Optional REST API: uvicorn api:app --reload  ->  POST /ask {"question": "..."}"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_eval.pipeline import RAGAssistant

app = FastAPI(title="RAG Eval Assistant")
bot = RAGAssistant.from_folder(os.environ.get("DOCS_DIR", "data/sample_docs"))


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(bot.chunks), "provider": bot.llm.name}


@app.post("/ask")
def ask(q: Question):
    return bot.ask(q.question).to_dict()
