"""Pluggable LLM providers. All expose: generate(system, user, contexts, question) -> str."""
from __future__ import annotations
import os
from .mock import MockLLM


def get_llm(name: str = "mock", model: str = ""):
    name = (name or "mock").lower()
    if name == "mock":
        return MockLLM()
    from . import remote
    classes = {"openai": remote.OpenAILLM, "anthropic": remote.AnthropicLLM, "claude": remote.AnthropicLLM,
               "gemini": remote.GeminiLLM, "groq": remote.GroqLLM, "ollama": remote.OllamaLLM}
    if name not in classes:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: mock, {', '.join(classes)}")
    return classes[name](model or None)
