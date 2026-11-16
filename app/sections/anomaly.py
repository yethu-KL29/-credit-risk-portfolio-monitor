"""Anomaly review tab (AI feature 2): review queue, reasons per flag, analyst decision."""
from __future__ import annotations

import streamlit as st
from common import Context, load_decisions, save_decision

EXPLAINER = """
- **What triggers it:** every account is scored by an Isolation Forest over (a) its repayment in the last 3 months vs its **own**
  first 9 months, and (b) robust z-scores vs its **peer segment** (same loan type x income band) for DPD, missed-payment rate,
  bureau utilisation, bureau DPD, credit-line count and external-debt-to-income. The top 3% most unusual are flagged; each flag lists
  the specific signals that exceeded thresholds (peer |z|≥3, own-history z≥2.5, ≥2 sudden misses, <50% EMI paid).
- **False positive:** a flagged account that is fine, e.g. a one-off bank holiday delay, a legitimate high-utilisation business
  borrower, or a small peer group making ordinary behaviour look extreme. Cost: wasted analyst time.
- **False negative:** a genuinely deteriorating account that is not flagged, e.g. slow, steady DPD creep that never breaks a
  threshold, or behaviour that is typical for its peers but still unsafe. Cost: missed early warning (partly covered by the PD model).
- **Human in the loop:** nothing is actioned automatically. The analyst reads the reasons, then records **Confirmed** (escalate to
  collections / credit review) or **Dismissed** (false positive) with a note; decisions are saved with reviewer and timestamp and
  become the labelled data for tuning thresholds.
"""


def _queue(ctx: Context):
    """Flagged accounts inside the sidebar filter, joined with any decision already recorded."""
    cols = ["account_id", "loan_type", "loan_amount", "pd_score", "risk_band"]
    q = ctx.flags.merge(ctx.data[cols], on="account_id")
    q = q[q.account_id.isin(ctx.view.account_id)]
    return q.merge(load_decisions()[["account_id", "decision"]], on="account_id", how="left").fillna({"decision": "Pending"})


def _decision_form(row) -> None:
    st.markdown(f"**Why {row.account_id} was flagged** (peer group: {row.peer_group}; anomaly score {row.anomaly_score:.3f})")
    for reason in row.reasons.split(" || "):
        st.markdown(f"- {reason}")
    st.caption(f"Context: model PD {row.pd_score:.1%} ({row.risk_band}); loan {row.loan_type}, ${row.loan_amount:,.0f}.")
    with st.form(f"decision_{row.account_id}"):
        choice = st.radio("Decision", ["Confirmed", "Dismissed"], horizontal=True)
        note = st.text_input("Note (optional)")
        who = st.text_input("Reviewer", "analyst")
        if st.form_submit_button("Save decision"):
            save_decision(row.account_id, choice, note, who)
            st.success(f"Saved: {row.account_id} -> {choice}")
            st.rerun()


def render(ctx: Context) -> None:
    st.subheader("Anomaly detection - review queue")
    with st.expander("How this AI feature works - trigger, false positive, false negative, human review"):
        st.markdown(EXPLAINER)
    fl = _queue(ctx)
    m = st.columns(4)
    m[0].metric("Flagged (in filter)", f"{len(fl):,}")
    for i, status in enumerate(["Pending", "Confirmed", "Dismissed"], start=1):
        m[i].metric(status, f"{(fl.decision == status).sum():,}")
    f1, f2 = st.columns(2)
    types = f1.multiselect("Flag type", sorted(fl.flag_type.unique()))
    statuses = f2.multiselect("Status", ["Pending", "Confirmed", "Dismissed"], default=["Pending"])
    q = fl[(fl.flag_type.isin(types) if types else True) & fl.decision.isin(statuses or ["Pending", "Confirmed", "Dismissed"])]
    st.dataframe(q[["account_id", "anomaly_score", "flag_type", "loan_type", "loan_amount", "pd_score", "decision"]].style.format(
        {"loan_amount": "${:,.0f}", "pd_score": "{:.1%}", "anomaly_score": "{:.3f}"}), hide_index=True, width="stretch", height=280)
    if q.empty:
        st.info("Nothing to review with these filters.")
    else:
        selected = st.selectbox("Review account", q.account_id.tolist())
        _decision_form(fl[fl.account_id == selected].iloc[0])
    st.caption("Decisions are written to data/processed/review_decisions.csv (audit trail in review_audit_log.csv). "
               "On free hosting the file system is ephemeral - production would write to a database.")
