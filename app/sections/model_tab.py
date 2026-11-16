"""Model & monitoring tab, and Data & method tab (static reporting of pipeline outputs)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import Context


def render_model(ctx: Context) -> None:
    m = ctx.metrics
    st.subheader("Default model - results")
    rows = [("Logistic regression (baseline)", "lr"), ("Gradient boosting (HistGB)", "gbm")]
    table = pd.DataFrame([{"Model": name, "AUC": m[k]["auc"], "KS": m[k]["ks"], "Gini": m[k]["gini"],
                           "5-fold CV AUC": m["cv_oof"][k]["auc"], "CV Gini": m["cv_oof"][k]["gini"]} for name, k in rows])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Hold-out: 30% stratified split ({m['n_test']:,} accounts); base default rate {m['default_rate']:.1%}. "
               "Signal is moderate (synthetic data): the tuned GBM is on par with the logistic baseline, so the baseline is a "
               "credible, more explainable champion; dashboard scores are out-of-fold GBM probabilities.")
    a, b = st.columns(2)
    imp = pd.DataFrame(m["importance"]).sort_values("auc_drop")
    a.plotly_chart(px.bar(imp, x="auc_drop", y="feature", orientation="h", height=420,
                          title="GBM permutation importance (AUC drop, hold-out)"), width="stretch")
    cal = pd.DataFrame(m["calibration"])
    fig = go.Figure()
    fig.add_scatter(x=cal.avg_pd, y=cal.actual, mode="lines+markers", name="Decile")
    fig.add_scatter(x=[0, .5], y=[0, .5], mode="lines", name="Perfect", line=dict(dash="dash"))
    fig.update_layout(title="Calibration by score decile (hold-out)", xaxis_title="Avg predicted PD",
                      yaxis_title="Actual default rate", height=420)
    b.plotly_chart(fig, width="stretch")
    st.markdown("**Logistic regression: largest standardised coefficients**")
    st.dataframe(pd.DataFrame(m["lr_coefficients"]).head(10), hide_index=True, width="stretch")

    mon = m["monitoring"]
    st.markdown("#### Monitoring after deployment")
    st.markdown(f"""
- **Stability:** PSI of the score and each key input, monthly, against the training baseline (alert >0.10, retrain review >0.25).
  Reference run, {mon['cohort_definition']}: score PSI = **{mon['score_psi_early_vs_late']}**; feature PSI: {mon['feature_psi']}.
- **Performance:** once outcomes mature (e.g. 12 months on book), track AUC / KS / Gini by vintage, calibration by decile and
  default rate by risk band; alert if AUC drops >0.03 or band default rates leave the calibration envelope.
- **Data quality:** daily checks on missing rates (interest rate, DPD), new categories, bureau feed latency.
- **Fairness / governance:** gender is not a model input (age is, so monitor age-band parity and get compliance sign-off); the model
  is used for prioritisation, not auto-decline; keep a model card and a champion-challenger setup with quarterly retraining.
- **Anomaly feed:** track the confirm/dismiss ratio of reviewed flags (precision proxy) and tune the 3% flag rate and z thresholds from it.
""")


def render_data(ctx: Context) -> None:
    st.subheader("Data ingestion, join logic and data-quality decisions")
    st.markdown("""
**Join logic (grain = loan account, 14,685 rows):** `loan_accounts` is the spine. `customers` joins many-to-one on `customer_id`; bureau
lines are aggregated to customer level then joined on `customer_id`; the 12 monthly repayment rows are aggregated to account level and
joined on `account_id`; collections events are aggregated to account level and joined on `account_id`. Aggregating before joining prevents
row fan-out; pandas `validate=` checks and an assertion on the row count guard this.

**Leakage control:** collections activity follows delinquency, so it is *not* a model feature. Repayment features use the 12-month
observation window only; bureau data is a customer-level snapshot.
""")
    st.dataframe(pd.DataFrame(ctx.dq), hide_index=True, width="stretch")
