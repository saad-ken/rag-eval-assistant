"""Deterministic offline 'LLM' so the app, tests and CI run with zero API keys.

It is *extractive*: it picks the context sentences that best overlap the
question and cites them. Not smart, but honest and reproducible — perfect
for testing the pipeline and the eval harness.
"""
from __future__ import annotations
import re
from ..prompts import REFUSAL
from ..text import tokenize


class MockLLM:
    name = "mock"

    def generate(self, system: str, user: str, contexts: list[str], question: str) -> str:
        q = set(tokenize(question))
        if not q or not contexts:
            return REFUSAL
        cands = []
        for ci, ctx in enumerate(contexts):
            ctx = re.sub(r"^\([^)]*\)\s*", "", ctx)  # drop the '(file > section)' label
            for sent in re.split(r"(?<=[.!?])\s+|\n+", ctx):
                s = sent.strip(" -*#")
                if len(s) < 15:
                    continue
                toks = set(tokenize(s))
                overlap = len(q & toks) / len(q)
                cands.append((overlap, -ci, s, ci))
        cands.sort(reverse=True)
        if not cands or cands[0][0] < 0.34:
            return REFUSAL
        best = [c for c in cands[:2] if c[0] >= max(0.34, cands[0][0] * 0.75)]
        return " ".join(f"{s.rstrip('.')}. [{ci + 1}]" for _, _, s, ci in best)
