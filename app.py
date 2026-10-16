"""Streamlit UI: upload your docs, ask questions, see cited sources, run the eval."""
import tempfile
from pathlib import Path
import streamlit as st
import yaml
from rag_eval.config import Settings
from rag_eval.evaluation import load_golden, run_eval
from rag_eval.loaders import load_dir, load_file
from rag_eval.pipeline import RAGAssistant

st.set_page_config(page_title="RAG Eval Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Eval Assistant")
st.caption("Answers only from your documents, cites every source, and says “I don't know” otherwise.")

s = Settings.from_env()
with st.sidebar:
    s.llm_provider = st.selectbox("LLM provider", ["mock", "openai", "anthropic", "gemini", "groq", "ollama"],
                                  index=0 if s.llm_provider == "mock" else
                                  ["mock", "openai", "anthropic", "gemini", "groq", "ollama"].index(s.llm_provider))
    s.top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, s.top_k)
    files = st.file_uploader("Upload PDF / MD / TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    st.info("No uploads? The sample Brightleaf handbook, product FAQ and test plan are used.")


@st.cache_resource(show_spinner="Indexing documents…")
def build(provider, top_k, names_and_bytes):
    cfg = Settings.from_env(); cfg.llm_provider, cfg.top_k = provider, top_k
    if names_and_bytes:
        tmp = Path(tempfile.mkdtemp())
        docs = []
        for name, data in names_and_bytes:
            (tmp / name).write_bytes(data)
            docs.append(load_file(tmp / name))
    else:
        docs = load_dir("data/sample_docs")
    return RAGAssistant(docs, cfg)


bot = build(s.llm_provider, s.top_k, tuple((f.name, f.getvalue()) for f in files or []))
tab_ask, tab_eval = st.tabs(["Ask", "Evaluate"])
with tab_ask:
    q = st.text_input("Your question", placeholder="How many days of annual leave do I get?")
    if q:
        ans = bot.ask(q)
        (st.warning if ans.refused else st.success)(ans.text)
        for i, h in enumerate(ans.hits, 1):
            with st.expander(f"[{i}] {h.chunk.label} — score {h.score:.2f}"):
                st.write(h.chunk.text)
with tab_eval:
    st.write("Runs the golden Q&A set (sample docs) and applies the quality gate.")
    if st.button("Run evaluation"):
        sample = RAGAssistant(load_dir("data/sample_docs"), s)
        rep = run_eval(sample, load_golden("evals/golden_set.yaml"),
                       yaml.safe_load(open("evals/thresholds.yaml")))
        st.subheader("✅ Quality gate passed" if rep["passed"] else "❌ Quality gate failed")
        cols = st.columns(len(rep["gate"]))
        for c, (m, g) in zip(cols, rep["gate"].items()):
            c.metric(m.replace("_", " "), f"{g['value']:.2f}", f"min {g['min']:.2f}", delta_color="off")
        st.dataframe([{k: r[k] for k in ("id", "passed", "question", "answer", "notes")} for r in rep["results"]],
                     use_container_width=True)
