# LEARN.md: understanding this project, step by step

This guide is for you, Saad. Read it top to bottom once, then run the commands as you go. By the end you should be able to explain every file and answer interview questions without notes.

---

## 1. The problem in one paragraph

LLMs like GPT or Claude don't know your company's documents, and when they don't know something they often **make it up confidently**. That's called *hallucination*. If an HR bot invents a leave policy, that's a real problem. This project does two things:

1. **The assistant** answers only from your documents, cites where each answer came from, and says "I don't know" otherwise.
2. **The eval suite** measures automatically whether the assistant actually behaves that way, and fails the build if quality drops.

Part 2 is where your QA background is the selling point. Most RAG demos have no tests at all.

---

## 2. What is RAG?

**RAG = Retrieval-Augmented Generation.** Instead of hoping the model "knows" the answer:

1. **Retrieve:** search your documents for the few passages most related to the question.
2. **Augment:** paste those passages into the prompt.
3. **Generate:** ask the LLM to answer *using only those passages*.

Analogy: it's an open-book exam. The model doesn't need to memorise the handbook. It just needs to find the right page and read it carefully.

Why not just paste every document into the prompt? Documents can be huge, cost goes up per token, and models get worse at finding facts in very long prompts.

---

## 3. Walking through the code (in the order a question flows)

### 3.1 Loading: `rag_eval/loaders.py`
Turns each file into plain text. PDFs use `pypdf`, while `.md` and `.txt` files are read directly. Each `Document` keeps its file name so we can cite it later.

### 3.2 Chunking: `rag_eval/chunking.py`
We split the text into **chunks** of about 600 characters, because:
- Search works better on small, focused pieces than on a whole 50-page file.
- We only send the best few chunks to the LLM, which keeps prompts short and cheap.

Two details matter:
- **Overlap (100 chars):** neighbouring chunks share a bit of text, so a fact that sits on a boundary isn't cut in half.
- **Section-aware:** we never mix two headings in one chunk, and each chunk remembers its heading. That's why a citation can say `handbook.md > Leave Policy` instead of just a file name.

Trade-off to mention in interviews: chunks that are too small lose context, and chunks that are too large bury the fact and waste tokens. Chunk size is a tunable setting, and **the eval suite is how you decide which value is best.**

### 3.3 Retrieval: `rag_eval/retrieval.py`
Given a question, find the top-k (default 4) most relevant chunks.

- **BM25 (default, offline).** A classic keyword-search formula, the same family Elasticsearch uses. It rewards chunks that contain the question's words, especially rare words ("parental" matters more than "employee"). We normalise the score to 0–1 so it can act as a *confidence*.
- **Embeddings (optional).** `sentence-transformers` or OpenAI turns each chunk into a vector, a list of numbers that captures meaning. Similar meanings end up close together, so "minimum password length" can match "at least 14 characters" even with no shared words. We compare vectors with **cosine similarity**.

BM25 is fast, free and explainable but misses synonyms. Embeddings understand meaning but need a model. Production systems often use **hybrid** search (both), which is on the roadmap.

### 3.4 Guardrail 1: refuse on weak retrieval (`pipeline.py`)
If the best chunk's score is below `MIN_SCORE` (0.15), we don't even call the LLM. We return "I don't know based on the provided documents." This is the cheapest anti-hallucination defence there is.

### 3.5 Prompting: `rag_eval/prompts.py`
The system prompt tells the model to:
1. Use only facts from the context.
2. Cite each sentence like `[1]`.
3. Say exactly "I don't know…" if the context doesn't answer the question.

The chunks are numbered `[1]`, `[2]`, … so the model's citations map back to real files. In `pipeline.py` we parse those `[n]` markers into source labels.

### 3.6 LLM providers: `rag_eval/providers/`
All providers share one method, `generate(...)`, so switching models is one environment variable (`LLM_PROVIDER=anthropic`). This is the **strategy / adapter pattern**.

The **mock provider** is important. It's a fake "LLM" that copies the most relevant sentences from the context. It's not clever, but it's **deterministic** (same input, same output) and needs no API key. That lets the unit tests and CI run free and reliably. Think of it like stubbing a payment gateway in test automation.

### 3.7 Interfaces
- `rag_eval/cli.py`: `python -m rag_eval ask "..."` and `python -m rag_eval eval`
- `app.py`: Streamlit web page where you upload docs, ask questions, see sources and run evals
- `api.py`: FastAPI `POST /ask` so other apps (or an n8n workflow!) can call it

---

## 4. The eval suite, and why each metric matters

**Golden set** (`evals/golden_set.yaml`): 25 hand-written test cases. That's exactly like test cases in a test plan, with an input, an expected result and an ID.

| Metric | QA analogy | What failure means |
|---|---|---|
| **Retrieval hit rate**: right file in the top-k? | "Did the test even reach the right page?" | If retrieval fails, no LLM can answer correctly. Fix chunking or search. |
| **Answer correctness**: required facts present? | Assertion on expected values | Wrong or incomplete answers |
| **Groundedness**: every sentence and number supported by retrieved text? | Checking the app didn't display data that isn't in the DB | **Hallucination.** The most dangerous failure. |
| **Citation accuracy**: cites the expected file? | Verifying the audit trail | Users can't verify the answer, or the citation points to the wrong place |
| **Refusal accuracy**: refuses only when it should? | Negative testing | Too few refusals means it invents answers. Too many means it's useless. |

**Quality gate** (`evals/thresholds.yaml`): the minimum score for each metric. Below the minimum, the eval exits with code 1 and GitHub Actions fails the build. It's the same idea as "the release ships only if the regression pass rate is ≥ 98%", which happens to be in our sample test plan too.

**Real story to tell:** while building this, the groundedness metric dropped to 0.90. The cause was that the mock model was leaking an internal label `(file > section)` into answers. The metric caught a genuine bug, I fixed it, and the score went to 1.00. That's the eval suite doing its job.

**Honest failures we kept** (from a real run: 22/25 cases fully pass):
- Synonym miss ("minimum" vs "at least"). This is a BM25 limitation.
- An incomplete answer (left out "14-day").
- One out-of-scope question that got answered instead of refused.

Keeping these visible is a good interview point: *"a test suite that always shows 100% probably isn't testing much."*

How the groundedness check works (simple version): split the answer into sentences, and require that ≥80% of each sentence's meaningful words and **all of its numbers** appear in the retrieved chunks. Numbers are strict because invented numbers ("30 days" instead of "24 days") are the classic hallucination. Its limit is that it checks words, not meaning. The upgrade is an **LLM-as-judge** (roadmap).

---

## 5. Tests and CI
- `tests/`: pytest unit tests for chunking, BM25 ranking, citations, refusals, determinism, the groundedness checker (including that it *catches* an invented number) and the quality gate (including that an impossible threshold *fails*).
- `.github/workflows/ci.yml`: on every push it runs `pytest` and then `python -m rag_eval eval`, both in mock mode, and uploads the HTML report as an artifact.

---

## 6. Explaining the project in an interview (≈60 seconds)

> "I built a RAG document assistant with a built-in evaluation suite. You upload docs like policies or test plans, and it answers with citations and refuses when the docs don't cover the question. It chunks documents by section, retrieves with BM25 or embeddings, and uses a grounded prompt with a pluggable LLM layer: OpenAI, Claude, Gemini, Groq or local Ollama.
> Coming from three years in QA automation, I treated it like any system under test. I wrote a golden set of 25 cases and five metrics: retrieval hit rate, correctness, groundedness, citation accuracy and refusal accuracy. A quality gate runs in GitHub Actions, so a prompt or model change that hurts quality fails the build. A deterministic mock LLM lets CI run without API keys. The groundedness metric actually caught a bug in my own pipeline during development."

---

## 7. Likely interview questions (with answers)

**Q: Why RAG instead of fine-tuning?**
RAG is cheaper and easier to keep up to date: change a document and the answer changes. It can also cite sources. Fine-tuning teaches style or behaviour, but it's a poor way to store changing facts.

**Q: How do you reduce hallucinations?**
Three layers: refuse when retrieval confidence is low, a strict prompt that says only use the context, cite and refuse otherwise, and measuring groundedness in evals so regressions are caught.

**Q: How did you choose chunk size and top-k?**
I started at 600 characters with 100 overlap and top-k of 4, then let the eval metrics decide. If retrieval hit rate is low, I'd change chunking or switch to embeddings. If correctness is low while retrieval is fine, the problem is the prompt or the model.

**Q: BM25 vs embeddings?**
BM25 matches keywords. It's fast, free, explainable and good with exact terms like error codes, but it misses synonyms. Embeddings match meaning. Hybrid search combines both. My eval showed a concrete BM25 synonym failure.

**Q: How do you evaluate an LLM app when outputs aren't deterministic?**
Use temperature 0, a golden set, and metrics that check properties (facts present, sources cited, no unsupported numbers) rather than exact strings. Run several trials for flaky cases, and use a deterministic mock for CI pipeline tests.

**Q: What are the limits of your groundedness metric?**
It's lexical, so it can miss a paraphrased hallucination or flag a valid paraphrase. Next I'd add an LLM-as-judge for faithfulness, calibrated against a few human-labelled examples.

**Q: How would you scale this?**
A persistent vector database (pgvector, Chroma), batched embedding at ingest time, caching, hybrid retrieval plus a reranker, and tracking eval scores per commit.

**Q: What's the mock provider for?**
It makes the pipeline testable: no cost, no keys, no network flakiness, identical output every run. It tests *my* code (retrieval, guardrails, citation parsing, eval harness), not the model.

**Q: What would you do if a stakeholder said "the bot gave a wrong answer"?**
Reproduce it, add it to the golden set as a new case (just like a bug turning into a regression test), find the failing stage with the metrics (retrieval or generation), fix it, and confirm the gate passes.

---

## 8. Try it yourself (15 minutes)
1. `python -m rag_eval ask "Which days are in-office collaboration days?"`, then look at the `[1]` citation.
2. `python -m rag_eval ask "Who is the CEO of Brightleaf Analytics?"`. Notice it *answers*. That's the known failure the eval reports.
3. Open `reports/eval_report.html` in a browser.
4. Edit `evals/thresholds.yaml`, set `answer_correctness: 0.95`, re-run `python -m rag_eval eval`, and watch the gate fail (exit code 1). Then set it back.
5. Add your own question to `golden_set.yaml` and re-run.
6. Put one of your own PDFs in a folder and run `python -m rag_eval --docs that_folder ask "..."`.
