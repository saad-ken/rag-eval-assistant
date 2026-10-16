from rag_eval.prompts import REFUSAL
from rag_eval.providers import get_llm
import pytest


def test_answer_has_citation(bot):
    ans = bot.ask("How many days of paid annual leave do full-time employees get?")
    assert "24 days" in ans.text
    assert not ans.refused
    assert ans.citations and ans.citations[0].startswith("handbook.md")


def test_refuses_out_of_scope(bot):
    ans = bot.ask("What is the best recipe for margherita pizza?")
    assert ans.refused and ans.text == REFUSAL


def test_mock_is_deterministic(bot):
    q = "Which browsers do the Playwright end-to-end tests run on?"
    assert bot.ask(q).text == bot.ask(q).text


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_llm("nope")


def test_to_dict_shape(bot):
    d = bot.ask("What are the core working hours?").to_dict()
    assert {"question", "answer", "refused", "citations", "sources"} <= d.keys()
