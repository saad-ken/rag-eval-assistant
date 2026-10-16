import yaml
from rag_eval.evaluation import groundedness_score, load_golden, run_eval, to_markdown, to_html


def test_groundedness_flags_invented_numbers():
    ctx = ["Employees receive 24 days of paid annual leave."]
    assert groundedness_score("Employees receive 24 days of paid annual leave. [1]", ctx) == 1.0
    assert groundedness_score("Employees receive 30 days of paid annual leave.", ctx) == 0.0


def test_groundedness_flags_invented_words():
    ctx = ["The Team plan costs 4,500 INR."]
    assert groundedness_score("The Team plan includes free unicorn rides and a yacht.", ctx) < 1


def test_golden_set_is_valid():
    cases = load_golden("evals/golden_set.yaml")
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c.get("out_of_scope") or (c.get("expected_source") and c.get("must_include"))


def test_quality_gate_passes_in_mock_mode(bot):
    rep = run_eval(bot, load_golden("evals/golden_set.yaml"), yaml.safe_load(open("evals/thresholds.yaml")))
    assert rep["passed"], rep["metrics"]
    assert "Quality gate" in to_html(rep) and "| Metric |" in to_markdown(rep)


def test_gate_fails_when_threshold_impossible(bot):
    rep = run_eval(bot, load_golden("evals/golden_set.yaml")[:3], {"answer_correctness": 1.01})
    assert not rep["passed"]
