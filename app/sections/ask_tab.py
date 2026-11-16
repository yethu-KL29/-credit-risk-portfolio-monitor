"""Ask the Portfolio tab (AI feature 1): natural-language Q&A grounded in the data (logic lives in src/ask.py)."""
from __future__ import annotations

import json

import plotly.express as px
import streamlit as st
from common import Context

from src import ask as ask_mod

EXPLAINER = """
- **What triggers it:** an analyst types a question. An LLM (Gemini/OpenAI, if an API key is set) or a rule-based parser converts it
  into a **whitelisted query spec** (metric, up to 2 group-by dimensions, filters). The LLM only sees the question plus field
  names / allowed values - **never customer rows**. Pandas computes the numbers on the full dataset; every answer shows the table and
  its source. LLM wording is accepted only if every number in it appears in the result table.
- **False positive (wrong answer given):** the question is mapped to the wrong metric/filter (e.g. "risky" read as default
  rate when the analyst meant predicted PD). Mitigated by showing the interpreted query spec and the full table.
- **False negative (wrongly declined):** a valid question is refused because it needs a field or phrasing the whitelist does not
  cover. Safe failure mode - the analyst rephrases or uses the other tabs.
- **Human in the loop:** the analyst checks the displayed interpretation and source table before relying on the answer; the
  assistant cannot take actions or change data. Forecasts, profit, PII and other out-of-scope questions are declined.
- **Scope:** always answers on the full portfolio (sidebar filters do not apply) unless the question itself contains filters.
"""
EXAMPLES = ["Which loan type has the highest default rate?", "Default rate by region",
            "What is the total exposure by loan type?", "Default rate by employment type and income band",
            "How many accounts were flagged by anomaly detection by loan type?", "What will the default rate be next year?"]


def render(ctx: Context) -> None:
    st.subheader("Ask the Portfolio")
    with st.expander("How this AI feature works - trigger, false positive, false negative, human review"):
        st.markdown(EXPLAINER)
    cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 3].button(example, key=f"ex{i}"):
            st.session_state["q"] = example
    question = st.text_input("Your question", key="q", placeholder="e.g. Which region has the lowest default rate for mortgages?")
    if not question:
        return
    with st.spinner("Thinking..."):
        res = ask_mod.ask(question, ctx.data)
    if not res.ok:
        st.warning(res.answer)
        return
    st.success(res.answer)
    st.markdown("**Result table**")
    st.dataframe(res.table, hide_index=True, width="stretch")
    st.caption(f"Source: {res.source}")
    st.caption(f"Interpreted as: `{json.dumps(res.spec)}` | parser: {res.mode}")
    if len(res.spec["group_by"]) == 1 and len(res.table) > 1:
        label = ask_mod.METRICS[res.spec["metric"]]
        st.plotly_chart(px.bar(res.table, x=res.spec["group_by"][0], y=label, height=300), width="stretch")
