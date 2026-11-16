import os

import pandas as pd
import pytest

from src import ask
from src.ingest import DATASET_PATH, PROC


@pytest.fixture(scope="module")
def data():
    os.environ.pop("OPENAI_API_KEY", None)
    df = pd.read_parquet(DATASET_PATH)
    return ask.prepare(df, pd.read_parquet(PROC / "scores.parquet"), pd.read_parquet(PROC / "anomaly_flags.parquet"))


def test_grain_and_no_fanout(data):
    assert data.account_id.is_unique and len(data) == 14685
    assert set(data.gender.unique()) == {"Male", "Female"}


def test_highest_default_loan_type_matches_pandas(data):
    r = ask.ask("Which loan type has the highest default rate?", data)
    assert r.ok
    truth = data.groupby("loan_type").target_default.mean().sort_values(ascending=False)
    assert truth.index[0] in r.answer and f"{truth.iloc[0]:.1%}" in r.answer


def test_filter_and_cross_cut(data):
    r = ask.ask("default rate by employment type and income band in the North region", data)
    assert r.ok and r.spec["filters"] == {"region": ["North"]} and len(r.spec["group_by"]) == 2


@pytest.mark.parametrize("q", ["What will the default rate be next year?", "Show me the CEO's phone number",
                               "What is our profit by region?", "blah blah"])
def test_declines(data, q):
    assert not ask.ask(q, data).ok


def test_spec_whitelist_rejects_unknown(data):
    with pytest.raises(ValueError):
        ask.validate_spec({"metric": "default_rate", "group_by": ["ssn"]}, data)


def test_hallucinated_numbers_rejected():
    t = pd.DataFrame({"accounts": [100], "default rate": [0.123]})
    assert ask._numbers_ok("Default rate is 12.3% over 100 accounts", t, "")
    assert not ask._numbers_ok("Default rate is 45.6%", t, "")


def test_provider_selection_prefers_gemini(monkeypatch):
    from src import llm
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ASK_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert not llm.llm_available()
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert llm.get_provider().name == "OpenAI"
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    p = llm.get_provider()
    assert p.name == "Gemini" and "generativelanguage" in p.base_url


def test_bad_llm_key_falls_back_to_rules(data, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")
    monkeypatch.setattr("src.llm.GEMINI_BASE_URL", "http://127.0.0.1:9/")  # unreachable -> forces failure
    r = ask.ask("Which loan type has the highest default rate?", data)
    assert r.ok and "Mortgage" in r.answer and r.mode == "rules"
