"""Portfolio tab: KPIs, risk segments, default rate by dimension, two-way drill-down and insights."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import AGE_ORDER, INCOME_ORDER, RISK_COLORS, RISK_ORDER, Context, money, pct

from src import ask as ask_mod

DIMENSIONS = {"Loan type": "loan_type", "Region": "region", "Employment type": "employment_type",
              "Age band": "age_band", "Income band": "income_band", "Origination year": "origination_year",
              "City tier": "city_tier", "Disbursal channel": "disbursal_channel"}


def _kpis(ctx: Context) -> None:
    v = ctx.view
    c = st.columns(5)
    c[0].metric("Accounts", f"{len(v):,}")
    c[1].metric("Exposure", money(v.loan_amount.sum()))
    c[2].metric("Default rate", pct(v.target_default.mean()),
                f"{(v.target_default.mean() - ctx.base_rate) * 100:+.1f} pp vs portfolio")
    c[3].metric("Defaulted exposure", money(v.defaulted_exposure.sum()))
    c[4].metric("Avg predicted PD", pct(v.pd_score.mean()))


def _risk_segments(ctx: Context, col) -> None:
    seg = ctx.view.groupby("risk_band").agg(accounts=("account_id", "size"), exposure=("loan_amount", "sum"),
                                            default_rate=("target_default", "mean")).reindex(RISK_ORDER).dropna().reset_index()
    col.subheader("Risk segments (model PD bands)")
    fig = px.bar(seg, x="risk_band", y="exposure", color="risk_band", color_discrete_map=RISK_COLORS,
                 hover_data={"accounts": True, "default_rate": ":.1%"})
    fig.update_layout(showlegend=False, height=330, yaxis_title="Exposure (USD)", xaxis_title=None)
    col.plotly_chart(fig, width="stretch")
    col.caption("Bands: Low <10%, Medium 10-20%, High 20-35%, Very High >35% out-of-fold predicted PD. "
                "Hover for realised default rate.")


def _ordered(g: pd.DataFrame, col: str) -> pd.DataFrame:
    """Sort a grouped frame sensibly: natural order for bands/years, worst-first otherwise."""
    if col == "age_band":
        return g.set_index(col).reindex(AGE_ORDER).dropna().reset_index()
    if col == "income_band":
        return g.set_index(col).reindex(INCOME_ORDER).dropna().reset_index()
    if col == "origination_year":
        return g.sort_values(col)
    return g.sort_values("default_rate", ascending=False)


def _default_by_dimension(ctx: Context, col) -> None:
    col.subheader("Default rate by dimension")
    label = col.selectbox("Dimension", list(DIMENSIONS), key="dim")
    dcol = DIMENSIONS[label]
    g = ctx.view.groupby(dcol).agg(accounts=("account_id", "size"), default_rate=("target_default", "mean"),
                                   exposure=("loan_amount", "sum")).reset_index()
    g[dcol] = g[dcol].astype(str)
    g = _ordered(g, dcol)
    fig = px.bar(g, x=dcol, y="default_rate", hover_data={"accounts": True, "exposure": ":,.0f"},
                 text=g.default_rate.map(pct))
    fig.add_hline(y=ctx.base_rate, line_dash="dash", annotation_text="portfolio avg")
    fig.update_layout(height=330, yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Default rate")
    col.plotly_chart(fig, width="stretch")


def _drilldown(ctx: Context) -> None:
    st.subheader("Drill-down: cross two dimensions")
    a, b, c = st.columns(3)
    d1 = a.selectbox("Rows", list(DIMENSIONS), index=2, key="d1")
    d2 = b.selectbox("Columns", list(DIMENSIONS), index=4, key="d2")
    measure = c.selectbox("Measure", ["Default rate", "Accounts", "Exposure"], key="mm")
    if d1 == d2:
        st.info("Pick two different dimensions.")
        return
    v = ctx.view
    piv = v.assign(_r=v[DIMENSIONS[d1]].astype(str), _c=v[DIMENSIONS[d2]].astype(str)).groupby(["_r", "_c"]).agg(
        accounts=("account_id", "size"), default_rate=("target_default", "mean"), exposure=("loan_amount", "sum")).reset_index()
    value = {"Default rate": "default_rate", "Accounts": "accounts", "Exposure": "exposure"}[measure]
    mat, n = piv.pivot(index="_r", columns="_c", values=value), piv.pivot(index="_r", columns="_c", values="accounts")
    if DIMENSIONS[d2] == "income_band":
        mat = mat.reindex(columns=[x for x in INCOME_ORDER if x in mat.columns])
        n = n.reindex(columns=mat.columns)
    fig = px.imshow(mat, text_auto=".1%" if value == "default_rate" else ",.0f", aspect="auto",
                    color_continuous_scale="RdYlGn_r" if value == "default_rate" else "Blues")
    fig.update_layout(height=380, xaxis_title=d2, yaxis_title=d1)
    st.plotly_chart(fig, width="stretch")
    small = int((n < ask_mod.MIN_GROUP).sum().sum())
    if small:
        st.caption(f"⚠ {small} cell(s) have fewer than {ask_mod.MIN_GROUP} accounts - treat as indicative.")


def _top_accounts(ctx: Context) -> None:
    st.subheader("Top-10 riskiest accounts in view")
    top = ctx.view.sort_values("pd_score", ascending=False).head(10)[
        ["account_id", "loan_type", "loan_amount", "employment_type", "pd_score", "risk_band", "driver1"]
    ].rename(columns={"driver1": "main driver"})
    st.dataframe(top.style.format({"loan_amount": "${:,.0f}", "pd_score": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("Copy an account_id into the Account view tab to see its key drivers.")


def _insights(ctx: Context) -> None:
    """Insights are computed from the full portfolio (not the filter) so the numbers are stable."""
    st.subheader("Portfolio insights (full portfolio) - with 'so what'")
    d, base = ctx.data, ctx.base_rate
    by_type = d.groupby("loan_type").agg(n=("account_id", "size"), exp=("loan_amount", "sum"),
                                         dr=("target_default", "mean"), dexp=("defaulted_exposure", "sum"))
    dr = lambda mask: d[mask].target_default.mean()  # noqa: E731
    low_inc, high_inc = dr(d.annual_income < 40000), dr(d.annual_income >= 100000)
    unemployed = dr(d.employment_type == "Unemployed")
    young_mortgage, over60 = dr((d.age < 30) & (d.loan_type == "Mortgage")), dr(d.age >= 60)
    miss0, miss1, miss2 = (dr(d.rep_missed_l3 == k) for k in (0, 1, 2))
    cross_bad = dr((d.employment_type == "Unemployed") & (d.annual_income < 60000))
    cross_good = dr((d.employment_type == "Salaried") & (d.annual_income >= 100000))
    region_dr = d.groupby("region").target_default.mean()
    m = by_type.loc["Mortgage"]
    st.markdown(f"""
1. **Mortgages are the concentration risk.** {m.n / by_type.n.sum():.0%} of accounts but {m.exp / by_type.exp.sum():.0%} of exposure
   and {m.dexp / by_type.dexp.sum():.0%} of defaulted exposure (default rate {pct(m.dr)} vs {pct(base)} overall).
   *So what:* tighten LTV / income checks and monitor mortgage vintages first.
2. **Income and employment drive default far more than geography.** Income <40k defaults at {pct(low_inc)} vs {pct(high_inc)} for 100k+;
   Unemployed {pct(unemployed)}. Regions only span {pct(region_dr.min())}-{pct(region_dr.max())}.
   *So what:* underwrite on affordability, not region quotas.
3. **Cross-cut: unemployed on <60k income default at {pct(cross_bad)} vs {pct(cross_good)} for salaried 100k+** ({cross_bad / cross_good:.1f}x),
   and under-30s with a mortgage reach {pct(young_mortgage)} vs {pct(over60)} for 60+.
   *So what:* cap exposure or require collateral / a co-signer in those cells.
4. **Recent missed payments are an early-warning signal.** 0 / 1 / 2 missed payments in the last 3 months -> default
   {pct(miss0)} / {pct(miss1)} / {pct(miss2)}. *So what:* start collections outreach at the first recent miss.
5. **Age is a clear gradient:** under-30s {pct(dr(d.age < 30))} vs {pct(over60)} for 60+.
   *So what:* price / limit young borrowers for thin-file risk rather than declining the segment.
""")


def render(ctx: Context) -> None:
    if ctx.view.empty:
        st.warning("No accounts match the current filters.")
        return
    _kpis(ctx)
    left, right = st.columns(2)
    _risk_segments(ctx, left)
    _default_by_dimension(ctx, right)
    _drilldown(ctx)
    _top_accounts(ctx)
    _insights(ctx)
