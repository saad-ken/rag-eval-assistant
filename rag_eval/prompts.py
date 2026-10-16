REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers ONLY from the provided context.
Rules:
1. Use only facts stated in the context. Never use outside knowledge.
2. After each sentence, cite the supporting source like [1] or [2].
3. If the context does not contain the answer, reply exactly: "{REFUSAL}"
4. Be concise: 1-4 sentences."""


def build_user_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
