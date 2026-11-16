"""Account view tab: one account's risk score, key drivers, repayment pattern, bureau lines, collections."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import Context, load_driver_deltas, load_raw_tables, pct

STATUS_COLORS = {"On-Time": "#2e9e5b", "Late": "#e0b000", "Missed": "#c62828"}


def _select_account(ctx: Context) -> str:
    pool = ctx.view.sort_values("pd_score", ascending=False)
    c1, c2 = st.columns([1, 2])
    typed = c1.text_input("Account ID (e.g. ACC200001)", "")
    picked = c2.selectbox("...or pick from the 200 riskiest accounts in the current filter",
                          [""] + pool.account_id.head(200).tolist())
    return typed.strip().upper() or picked


def _score_header(a: pd.Series, base_rate: float) -> None:
    k = st.columns(4)
    k[0].metric("Predicted default probability", pct(a.pd_score), f"{(a.pd_score - base_rate) * 100:+.1f} pp vs portfolio avg")
    k[1].metric("Risk band", a.risk_band)
    k[2].metric("Actual outcome (label)", "Defaulted" if a.target_default == 1 else "Performing")
    k[3].metric("Anomaly flag", "FLAGGED" if a.anomaly_flag else "No")


def _drivers_chart(a: pd.Series, col) -> None:
    col.markdown("**Key drivers of this score** (change in predicted PD when the factor group is set to the portfolio "
                 "median/mode; red = pushes risk up)")
    d = load_driver_deltas().loc[a.account_id].sort_values()
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h",
                           marker_color=["#c62828" if v > 0 else "#2e9e5b" for v in d.values]))
    fig.update_layout(height=380, xaxis_title="Δ predicted PD (probability points)", margin=dict(l=0, r=0, t=10, b=0))
    col.plotly_chart(fig, width="stretch")
    col.markdown(f"Top risk drivers: **{a.driver1}**, **{a.driver2}**, **{a.driver3}**")


def _profile_table(a: pd.Series, col) -> None:
    col.markdown("**Profile & loan**")
    years = a.years_employed if pd.notna(a.years_employed) else "unknown"
    rate = a.interest_rate_pct if pd.notna(a.interest_rate_pct) else "missing"
    rows = {
        "Customer": a.customer_id, "Age / gender": f"{a.age} / {a.gender}",
        "Employment": f"{a.employment_type} ({years} yrs)", "Annual income": f"${a.annual_income:,.0f}",
        "Region / tier": f"{a.region} / {a.city_tier}", "Loan type": a.loan_type, "Loan amount": f"${a.loan_amount:,.0f}",
        "Rate / tenure": f"{rate}% / {a.tenure_months}m", "Originated": str(a.origination_date.date()),
        "Bureau lines / write-offs": f"{int(a.bureau_lines)} / {int(a.bureau_writeoffs)}",
        "Bureau utilisation": f"{a.bureau_utilisation:.0%}",
    }
    col.dataframe(pd.DataFrame({"Field": rows.keys(), "Value": rows.values()}), hide_index=True, width="stretch")


def _repayment_chart(account_id: str) -> None:
    repay, _, _ = load_raw_tables()
    r = repay[repay.account_id == account_id]
    fig = go.Figure()
    fig.add_bar(x=r.payment_month, y=r.days_past_due, name="Days past due",
                marker_color=r.payment_status.map(STATUS_COLORS))
    fig.update_layout(height=260, title="Monthly repayment behaviour (green on-time, amber late, red missed)",
                      xaxis_title="Month since origination", yaxis_title="DPD", margin=dict(t=40, b=0))
    st.plotly_chart(fig, width="stretch")


def _bureau_and_collections(a: pd.Series) -> None:
    _, bureau, coll = load_raw_tables()
    c1, c2 = st.columns(2)
    c1.markdown("**External bureau lines**")
    c1.dataframe(bureau[bureau.customer_id == a.customer_id][
        ["credit_type", "lender_category", "sanctioned_amount", "outstanding_amount", "account_status", "days_past_due"]],
        hide_index=True, width="stretch")
    c2.markdown("**Collections events**")
    events = coll[coll.account_id == a.account_id]
    c2.dataframe(events[["event_date", "action_type", "outcome", "amount_recovered"]] if len(events)
                 else pd.DataFrame({"info": ["None"]}), hide_index=True, width="stretch")


def render(ctx: Context) -> None:
    st.subheader("Account-level risk score and key drivers")
    account_id = _select_account(ctx)
    if not account_id:
        st.info("Select or type an account ID.")
        return
    if account_id not in set(ctx.data.account_id):
        st.error("Account not found.")
        return
    a = ctx.data[ctx.data.account_id == account_id].iloc[0]
    _score_header(a, ctx.base_rate)
    left, right = st.columns(2)
    _drivers_chart(a, left)
    _profile_table(a, right)
    _repayment_chart(account_id)
    _bureau_and_collections(a)
    if a.anomaly_flag:
        st.warning("This account is flagged - see the Anomaly review tab for reasons and to record a decision.")
