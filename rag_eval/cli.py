"""Command line: `python -m rag_eval ask ...` and `python -m rag_eval eval`."""
from __future__ import annotations
import argparse
import json
import sys
import yaml
from .config import Settings
from .evaluation import load_golden, run_eval, write_reports
from .pipeline import RAGAssistant


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag_eval", description="Cited document Q&A + RAG quality gate")
    p.add_argument("--docs", default="data/sample_docs", help="folder with PDF/MD/TXT files")
    p.add_argument("--provider", help="override LLM_PROVIDER (mock, openai, anthropic, gemini, groq, ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask one question (or start a chat with no question)")
    a.add_argument("question", nargs="*")
    a.add_argument("--json", action="store_true")
    ev = sub.add_parser("eval", help="run the golden set and enforce the quality gate")
    ev.add_argument("--golden", default="evals/golden_set.yaml")
    ev.add_argument("--thresholds", default="evals/thresholds.yaml")
    ev.add_argument("--out", default="reports")
    args = p.parse_args(argv)

    s = Settings.from_env()
    if args.provider:
        s.llm_provider = args.provider
    bot = RAGAssistant.from_folder(args.docs, s)

    if args.cmd == "ask":
        def show(q):
            ans = bot.ask(q)
            if args.json:
                print(json.dumps(ans.to_dict(), indent=2))
                return
            print(f"\n{ans.text}\n")
            for i, h in enumerate(ans.hits, 1):
                print(f"  [{i}] {h.chunk.label}  (score {h.score:.2f})")
        if args.question:
            show(" ".join(args.question))
        else:
            print(f"Loaded {len(bot.chunks)} chunks. Ask a question (Ctrl+C to quit).")
            try:
                while q := input("\n? ").strip():
                    show(q)
            except (EOFError, KeyboardInterrupt):
                pass
        return 0

    thresholds = yaml.safe_load(open(args.thresholds, encoding="utf-8"))
    rep = run_eval(bot, load_golden(args.golden), thresholds)
    paths = write_reports(rep, args.out)
    print(f"Provider={rep['provider']} retriever={rep['retriever']} cases={rep['num_cases']} "
          f"fully_passed={rep['cases_passed']}")
    for m, g in rep["gate"].items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {m:<20} {g['value']:.2f}  (min {g['min']:.2f})")
    print(f"Quality gate: {'PASSED' if rep['passed'] else 'FAILED'}  -> {paths['html']}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
