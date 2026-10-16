"""Evaluation harness: run a golden Q&A set through the assistant and score it.

Metrics (all 0..1, higher is better):
- retrieval_hit_rate  : in-scope questions where the expected source file is in the top-k chunks
- answer_correctness  : in-scope questions whose answer contains every required key fact
- groundedness        : share of answer sentences (and numbers) supported by retrieved text
                        -> catches made-up facts / hallucinations
- citation_accuracy   : answered in-scope questions that cite the expected source
- refusal_accuracy    : questions where the assistant refused exactly when it should
                        (out-of-scope => "I don't know"; in-scope => answer)
"""
from __future__ import annotations
import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml
from .pipeline import RAGAssistant
from .text import tokenize

METRICS = ["retrieval_hit_rate", "answer_correctness", "groundedness", "citation_accuracy", "refusal_accuracy"]


@dataclass
class CaseResult:
    id: str
    question: str
    out_of_scope: bool
    answer: str
    refused: bool
    retrieved: list[str]
    cited: list[str]
    retrieval_hit: bool | None = None
    correct: bool | None = None
    groundedness: float | None = None
    citation_ok: bool | None = None
    refusal_ok: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.out_of_scope:
            return self.refusal_ok
        return bool(self.retrieval_hit and self.correct and self.citation_ok and (self.groundedness or 0) >= 0.99)


def groundedness_score(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences whose content words AND numbers appear in the context."""
    ctx = " ".join(contexts).lower()
    ctx_tokens = set(tokenize(ctx))
    ctx_numbers = set(re.findall(r"\d[\d,.:]*", ctx))
    clean = re.sub(r"\[\d+\]", "", answer)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", clean) if len(tokenize(s)) > 0]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        toks = tokenize(s)
        nums = [n.rstrip(".,") for n in re.findall(r"\d[\d,.:]*", s)]
        word_support = sum(t in ctx_tokens for t in toks) / len(toks)
        nums_ok = all(n in ctx_numbers or n.rstrip(".") in ctx for n in nums)
        ok += word_support >= 0.8 and nums_ok
    return ok / len(sents)


def load_golden(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def evaluate_case(assistant: RAGAssistant, case: dict) -> CaseResult:
    ans = assistant.ask(case["question"])
    oos = bool(case.get("out_of_scope"))
    retrieved = [h.chunk.source for h in ans.hits]
    r = CaseResult(case["id"], case["question"], oos, ans.text, ans.refused,
                   [h.chunk.label for h in ans.hits], ans.citations)
    r.refusal_ok = ans.refused == oos
    if not ans.refused:
        r.groundedness = groundedness_score(ans.text, [h.chunk.text for h in ans.hits])
        if r.groundedness < 1:
            r.notes.append("answer contains unsupported content")
    if oos:
        if not ans.refused:
            r.notes.append("answered an out-of-scope question")
        return r
    exp = case["expected_source"]
    r.retrieval_hit = exp in retrieved
    low = ans.text.lower()
    missing = [f for f in case.get("must_include", []) if f.lower() not in low]
    r.correct = not ans.refused and not missing
    if missing:
        r.notes.append(f"missing facts: {missing}")
    if ans.refused:
        r.notes.append("refused an in-scope question")
        r.citation_ok = False
    else:
        r.citation_ok = any(c.split(" > ")[0] == exp for c in ans.citations)
        if not r.citation_ok:
            r.notes.append("did not cite expected source")
    return r


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 1.0


def summarize(results: list[CaseResult]) -> dict:
    ins = [r for r in results if not r.out_of_scope]
    answered_in = [r for r in ins if not r.refused]
    return {
        "retrieval_hit_rate": _mean(r.retrieval_hit for r in ins),
        "answer_correctness": _mean(r.correct for r in ins),
        "groundedness": _mean(r.groundedness for r in results if r.groundedness is not None),
        "citation_accuracy": _mean(r.citation_ok for r in answered_in),
        "refusal_accuracy": _mean(r.refusal_ok for r in results),
    }


def run_eval(assistant: RAGAssistant, golden: list[dict], thresholds: dict) -> dict:
    t0 = time.time()
    results = [evaluate_case(assistant, c) for c in golden]
    metrics = summarize(results)
    gate = {m: {"value": metrics[m], "min": thresholds.get(m, 0), "pass": metrics[m] >= thresholds.get(m, 0)}
            for m in METRICS}
    return {
        "provider": getattr(assistant.llm, "name", "?"),
        "retriever": assistant.retriever.name,
        "num_cases": len(results),
        "num_chunks": len(assistant.chunks),
        "cases_passed": sum(r.passed for r in results),
        "seconds": round(time.time() - t0, 2),
        "metrics": metrics,
        "gate": gate,
        "passed": all(g["pass"] for g in gate.values()),
        "results": [asdict(r) | {"passed": r.passed} for r in results],
    }


def to_markdown(rep: dict) -> str:
    lines = [f"# RAG Evaluation Report — {'✅ PASS' if rep['passed'] else '❌ FAIL'}", "",
             f"Provider: `{rep['provider']}` · Retriever: `{rep['retriever']}` · Cases: {rep['num_cases']} "
             f"· Chunks: {rep['num_chunks']} · Cases fully passed: {rep['cases_passed']}/{rep['num_cases']}", "",
             "| Metric | Score | Min | Status |", "|---|---|---|---|"]
    for m, g in rep["gate"].items():
        lines.append(f"| {m} | {g['value']:.2f} | {g['min']:.2f} | {'✅' if g['pass'] else '❌'} |")
    lines += ["", "## Cases", "", "| ID | Pass | Answer | Notes |", "|---|---|---|---|"]
    for r in rep["results"]:
        a = r["answer"].replace("|", "\\|").replace("\n", " ")
        a = a[:140] + ("…" if len(a) > 140 else "")
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {a} | {'; '.join(r['notes'])} |")
    return "\n".join(lines) + "\n"


def to_html(rep: dict) -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(m)}</td><td>{g['value']:.2f}</td><td>{g['min']:.2f}</td>"
        f"<td class={'ok' if g['pass'] else 'bad'}>{'PASS' if g['pass'] else 'FAIL'}</td></tr>"
        for m, g in rep["gate"].items())
    cases = "".join(
        f"<tr><td>{e(r['id'])}</td><td class={'ok' if r['passed'] else 'bad'}>{'✔' if r['passed'] else '✘'}</td>"
        f"<td>{e(r['question'])}</td><td>{e(r['answer'])}</td><td>{e(', '.join(r['cited']))}</td>"
        f"<td>{e('; '.join(r['notes']))}</td></tr>" for r in rep["results"])
    status = "PASS" if rep["passed"] else "FAIL"
    return f"""<!doctype html><html><head><meta charset=utf-8><title>RAG Eval Report</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:14px}}th{{background:#f4f4f4}}
.ok{{color:#0a7d2c;font-weight:600}}.bad{{color:#c0262d;font-weight:600}}</style></head><body>
<h1>RAG Evaluation Report — <span class={'ok' if rep['passed'] else 'bad'}>{status}</span></h1>
<p>Provider <b>{e(rep['provider'])}</b> · Retriever <b>{e(rep['retriever'])}</b> · {rep['num_cases']} cases ·
{rep['cases_passed']} fully passed · {rep['seconds']}s</p>
<h2>Quality gate</h2><table><tr><th>Metric</th><th>Score</th><th>Min</th><th>Status</th></tr>{rows}</table>
<h2>Cases</h2><table><tr><th>ID</th><th>Pass</th><th>Question</th><th>Answer</th><th>Cited</th><th>Notes</th></tr>{cases}</table>
</body></html>"""


def write_reports(rep: dict, out_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "eval_report.json", "md": out / "eval_report.md", "html": out / "eval_report.html"}
    paths["json"].write_text(json.dumps(rep, indent=2), encoding="utf-8")
    paths["md"].write_text(to_markdown(rep), encoding="utf-8")
    paths["html"].write_text(to_html(rep), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}
