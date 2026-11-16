"""Dashboard entry point:  streamlit run app/app.py

Layout: sidebar filters -> six tabs. Each tab lives in app/sections/, shared helpers in app/common.py,
and all data/model/AI logic in src/ (the app only displays what src/ produced).
"""
import streamlit as st

st.set_page_config(page_title="Credit Risk Portfolio", page_icon="📊", layout="wide")

import common  # noqa: E402
from common import RISK_ORDER, Context  # noqa: E402
from sections import account, anomaly, ask_tab, model_tab, portfolio  # noqa: E402

from src import llm  # noqa: E402

common.load_secrets_into_env()  # must run before anything calls the LLM
data, flags, metrics, dq = common.load_all()

# ---- sidebar filters (drive the Portfolio, Account and Anomaly tabs) ----
st.sidebar.title("Filters")
st.sidebar.caption("Apply to the Portfolio, Account and Anomaly tabs.")
FILTERS = [("Loan type", "loan_type", sorted(data.loan_type.unique())),
           ("Region", "region", sorted(data.region.unique())),
           ("Employment type", "employment_type", sorted(data.employment_type.unique())),
           ("Income band", "income_band", common.INCOME_ORDER),
           ("Model risk band", "risk_band", RISK_ORDER)]
view = data
for label, column, options in FILTERS:
    chosen = st.sidebar.multiselect(label, options)
    if chosen:
        view = view[view[column].isin(chosen)]
st.sidebar.metric("Accounts in view", f"{len(view):,} / {len(data):,}")
st.sidebar.caption(f"LLM mode: {llm.describe()}")

ctx = Context(data=data, view=view, flags=flags, metrics=metrics, dq=dq, base_rate=data.target_default.mean())

# ---- tabs ----
st.title("Credit Risk Portfolio Monitor")
tabs = st.tabs(["📊 Portfolio", "🔎 Account view", "🚩 Anomaly review", "💬 Ask the Portfolio",
                "📈 Model & monitoring", "🧹 Data & method"])
renderers = [portfolio.render, account.render, anomaly.render, ask_tab.render, model_tab.render_model, model_tab.render_data]
for tab, render in zip(tabs, renderers):
    with tab:
        render(ctx)
