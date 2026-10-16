"""Real LLM providers. SDKs are imported lazily so you only install what you use."""
from __future__ import annotations
import json
import os
import urllib.request


class OpenAILLM:
    name = "openai"

    def __init__(self, model=None):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model or "gpt-4o-mini"

    def generate(self, system, user, contexts, question):
        r = self.client.chat.completions.create(model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content.strip()


class GroqLLM(OpenAILLM):
    name = "groq"

    def __init__(self, model=None):
        from groq import Groq
        self.client, self.model = Groq(), model or "llama-3.1-8b-instant"


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model=None):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model or "claude-3-5-haiku-latest"

    def generate(self, system, user, contexts, question):
        r = self.client.messages.create(model=self.model, max_tokens=512, temperature=0,
                                        system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


class GeminiLLM:
    name = "gemini"

    def __init__(self, model=None):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model or "gemini-1.5-flash")

    def generate(self, system, user, contexts, question):
        return self.model.generate_content(system + "\n\n" + user).text.strip()


class OllamaLLM:
    """Local models via Ollama's HTTP API (no SDK needed)."""
    name = "ollama"

    def __init__(self, model=None):
        self.model = model or "llama3.1"
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, system, user, contexts, question):
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"].strip()
