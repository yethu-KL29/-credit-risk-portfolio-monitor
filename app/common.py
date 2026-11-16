"""Shared helpers for the dashboard: data loading, formatting, review-decision storage."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import ask as ask_mod  # noqa: E402
from src import llm  # noqa: E402
from src.ingest import DATASET_PATH, DQ_PATH, PROC, RAW  # noqa: E402

DECISIONS_PATH = PROC / "review_decisions.csv"
AUDIT_PATH = PROC / "review_audit_log.csv"
RISK_ORDER = ["Low", "Medium", "High", "Very High"]
RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#e0b000", "High": "#e8742a", "Very High": "#c62828"}
INCOME_ORDER = ["<40k", "40-60k", "60-80k", "80-100k", "100k+"]
AGE_ORDER = ["<30", "30-39", "40-49", "50-59", "60+"]


@dataclass
class Context:
    """Everything a dashboard section needs. `view` is `data` after the sidebar filters."""
    data: pd.DataFrame
    view: pd.DataFrame
    flags: pd.DataFrame
    metrics: dict
    dq: list
    base_rate: float


def load_secrets_into_env() -> None:
    """On Streamlit Cloud the API key lives in st.secrets; expose it as env vars for src/llm.py."""
    try:
        for name in llm.SECRET_NAMES:
            if name in st.secrets and not os.getenv(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:  # no secrets file locally - that is fine
        pass


@st.cache_data(show_spinner="Building data and model for first run (about a minute)...")
def load_all():
    needed = [DATASET_PATH, PROC / "scores.parquet", PROC / "anomaly_flags.parquet", PROC / "model_metrics.json"]
    if not all(p.exists() for p in needed):
        from src import build
        build.main()
    df = pd.read_parquet(DATASET_PATH)
    scores = pd.read_parquet(PROC / "scores.parquet")
    flags = pd.read_parquet(PROC / "anomaly_flags.parquet")
    metrics = json.loads((PROC / "model_metrics.json").read_text())
    dq = json.loads(DQ_PATH.read_text())
    data = ask_mod.prepare(df, scores, flags)
    driver_cols = [c for c in scores.columns if c.startswith("driver")]
    data = data.merge(scores[["account_id", *driver_cols]], on="account_id", how="left")
    data["defaulted_exposure"] = data.loan_amount * data.target_default
    return data, flags, metrics, dq


@st.cache_data
def load_raw_tables():
    """Raw tables used only for the single-account drill-down charts."""
    return (pd.read_csv(RAW / "repayment_history.csv"), pd.read_csv(RAW / "bureau_credit.csv"),
            pd.read_csv(RAW / "collections_events.csv"))


@st.cache_data
def load_driver_deltas() -> pd.DataFrame:
    return pd.read_parquet(PROC / "driver_deltas.parquet").set_index("account_id")


# ---- analyst decisions on anomaly flags (human-in-the-loop) -------------------------------------------
def load_decisions() -> pd.DataFrame:
    if DECISIONS_PATH.exists():
        return pd.read_csv(DECISIONS_PATH)
    return pd.DataFrame(columns=["account_id", "decision", "note", "reviewer", "timestamp"])


def save_decision(account_id: str, decision: str, note: str, reviewer: str) -> None:
    """Latest decision per account goes to review_decisions.csv; every decision is appended to the audit log."""
    current = load_decisions()
    current = current[current.account_id != account_id]
    row = pd.DataFrame([{"account_id": account_id, "decision": decision, "note": note, "reviewer": reviewer,
                         "timestamp": datetime.now().isoformat(timespec="seconds")}])
    pd.concat([current, row]).to_csv(DECISIONS_PATH, index=False)
    row.to_csv(AUDIT_PATH, mode="a", header=not AUDIT_PATH.exists(), index=False)


# ---- formatting ----------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{x:.1%}"


def money(x: float) -> str:
    return f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x:,.0f}"
