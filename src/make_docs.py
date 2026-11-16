"""Generate docs/architecture.pdf and docs/solution_overview.pdf (one page each): python -m src.make_docs"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from .ingest import PROC, ROOT  # noqa: E402

DOCS = ROOT / "docs"


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.05, body, ha="center", va="top", fontsize=6.6, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label=None, color="#333", style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, ha="center", fontsize=6.2, color=color, style="italic")


def architecture():
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.965, "Credit Risk Portfolio Monitor - Architecture", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.94, "raw data -> ingestion -> analytics -> model -> dashboard, with both AI features", ha="center", fontsize=8.5, color="#555")
    blue, green, yellow, red, grey = "#dbe9f9", "#dff3e3", "#fff4cc", "#fde0dc", "#eeeeee"
    # Row 1: pipeline
    box(ax, 0.02, 0.66, 0.17, 0.24, "1. RAW DATA  data/raw/", "customers.csv (10k)\nloan_accounts.csv (14.7k)\nbureau_credit.csv (23k)\n"
        "repayment_history.csv (176k)\ncollections_events.csv (8.6k)\n\nBank source systems in prod", grey)
    box(ax, 0.22, 0.66, 0.18, 0.24, "2. INGESTION  src/ingest.py", "pandas 2.3 + pyarrow\naudit DQ issues -> dq_log.json\nstandardise gender, flag/impute nulls\n"
        "aggregate bureau (cust) & repayment\n(acct) BEFORE joining (no fan-out)\nLEFT-join onto loan spine\n-> account_dataset.parquet", blue)
    box(ax, 0.43, 0.66, 0.17, 0.24, "3. ANALYTICS", "pandas group-bys / cross-tabs\ndefault rate, exposure by income,\nemployment, region, age, loan type\ninsights + 'so what'\n(computed live in the app)", blue)
    box(ax, 0.63, 0.66, 0.17, 0.24, "4. MODEL  src/model.py", "scikit-learn: Logistic Regression\n(baseline) + HistGradientBoosting\n41 engineered bureau / repayment /\nloan features; AUC, KS, Gini\npermutation importance; OOF PD\nper-account occlusion drivers", green)
    box(ax, 0.83, 0.66, 0.15, 0.24, "5. ANOMALY  src/anomaly.py", "IsolationForest + robust\nz-scores vs peer segment and\nown-history z (last 3 vs first 9\nmonths); reasons per flag\n-> anomaly_flags.parquet", yellow)
    for x1, x2 in [(0.19, 0.22), (0.40, 0.43), (0.60, 0.63), (0.80, 0.83)]:
        arrow(ax, x1, 0.78, x2, 0.78)
    # Row 2: dashboard
    ax.add_patch(FancyBboxPatch((0.02, 0.07), 0.96, 0.46, boxstyle="round,pad=0.01,rounding_size=0.012", fc="#f7f7fb", ec="#333", lw=1.4))
    ax.text(0.5, 0.515, "6. DASHBOARD  app/app.py  (Streamlit + Plotly, hosted on Streamlit Community Cloud)", ha="center", fontsize=10, fontweight="bold")
    box(ax, 0.04, 0.30, 0.20, 0.18, "Portfolio view", "KPIs, risk-band exposure,\ndefault rate by dimension,\ncross-dimension drill-down,\nsidebar filters", blue)
    box(ax, 0.26, 0.30, 0.20, 0.18, "Account view", "PD score, risk band,\nkey drivers, repayment\nchart, bureau lines,\ncollections events", green)
    box(ax, 0.50, 0.30, 0.23, 0.18, "AI #2 Anomaly review", "review queue + reasons;\nanalyst CONFIRMS / DISMISSES\n-> review_decisions.csv\n(human-in-the-loop, audit log)", yellow)
    box(ax, 0.75, 0.30, 0.21, 0.18, "Model & Data tabs", "metrics, importance,\ncalibration, PSI monitoring,\nDQ log, join logic", grey)
    # AI #1 flow
    box(ax, 0.04, 0.09, 0.21, 0.17, "AI #1 Ask the Portfolio (UI)", "analyst types a question;\nanswer + result table + source\nline + interpreted query shown", red)
    box(ax, 0.30, 0.09, 0.22, 0.17, "src/ask.py  parser", "LLM -> JSON query spec, or\nrule-based parser if no API key\nspec validated vs WHITELIST;\nout-of-scope questions DECLINED", red)
    box(ax, 0.57, 0.09, 0.19, 0.17, "pandas executor", "computes figures on the full\naccount_dataset (no LLM maths)\nn<30 groups flagged", blue)
    box(ax, 0.80, 0.09, 0.17, 0.17, "LLM phrasing check", "text kept only if every number\nis in the result table, else\ntemplate answer used", red)
    for x1, x2 in [(0.25, 0.30), (0.52, 0.57), (0.76, 0.80)]:
        arrow(ax, x1, 0.175, x2, 0.175)
    for x in (0.14, 0.36, 0.615, 0.855):
        arrow(ax, x, 0.66, x, 0.485, color="#999") if False else None
    arrow(ax, 0.31, 0.66, 0.15, 0.48, color="#2a6", style="-|>")
    arrow(ax, 0.715, 0.66, 0.36, 0.48, color="#2a6")
    arrow(ax, 0.905, 0.66, 0.62, 0.48, color="#2a6")
    ax.text(0.5, 0.57, "parquet / json artifacts feed every tab", ha="center", fontsize=6.5, color="#2a6", style="italic")
    # LLM callout
    ax.add_patch(FancyBboxPatch((0.02, 0.005), 0.96, 0.06, boxstyle="round,pad=0.005,rounding_size=0.01", fc="#fff", ec="#c62828", lw=1.2))
    ax.text(0.5, 0.035, "WHERE THE LLM IS CALLED (src/ask.py + src/llm.py only; Google Gemini or OpenAI API, key in Streamlit secrets / env):\n"
            "call 1 = question text + field names + allowed category values -> JSON query spec.   call 2 = the small AGGREGATED result table (<=10 rows) -> wording.\n"
            "NEVER sent: customer rows, IDs, raw tables, anomaly/model internals.  No key set -> LLM skipped, rule-based parser answers.  Anomaly model uses no LLM.",
            ha="center", va="center", fontsize=6.7)
    fig.savefig(DOCS / "architecture.pdf")
    plt.close(fig)


def solution():
    m = json.loads((PROC / "model_metrics.json").read_text())
    S = lambda **k: ParagraphStyle("s", fontName="Helvetica", fontSize=8.4, leading=10.2, **k)  # noqa: E731
    H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8.6, leading=10, spaceBefore=3.5, spaceAfter=1, textColor=colors.HexColor("#1a3c6e"))
    T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=1)
    doc = SimpleDocTemplate(str(DOCS / "solution_overview.pdf"), pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=9 * mm, bottomMargin=8 * mm)
    P = lambda t: Paragraph(t, S())  # noqa: E731
    lr, gb, cv = m["lr"], m["gbm"], m["cv_oof"]
    story = [
        Paragraph("Credit Risk Portfolio Monitor - Solution Overview", T),
        P("Streamlit dashboard over 5 synthetic bank tables (14,685 loan accounts, 16.1% default rate) with two embedded AI features. "
          "Pipeline: <b>raw CSV -> pandas ingestion -> analytics -> scikit-learn model -> Streamlit/Plotly</b> (see architecture.pdf)."),
        Paragraph("What the dashboard shows - view by view", H),
        P("<b>Portfolio:</b> KPIs (accounts, exposure, default rate, defaulted exposure, avg PD); exposure by model risk band; default rate by income, "
          "employment, region, age, loan type, vintage, channel; a two-dimension heat-map drill-down; sidebar filters (loan type, region, employment, income, "
          "risk band) re-cut every view. Analyst use: spot concentrations, then drill into the worst cell. "
          "<b>Account view:</b> enter an ID or pick from the 200 riskiest: predicted PD vs portfolio, risk band, top drivers (occlusion: score change when a factor "
          "group is set to the portfolio norm), 12-month repayment chart, bureau lines, collections history. Analyst use: explain why an account is risky before acting. "
          "<b>Anomaly review</b> and <b>Ask the Portfolio</b> tabs hold the AI features; <b>Model</b> tab: metrics, importance, calibration, monitoring; <b>Data</b> tab: join logic and every data-quality decision."),
        Paragraph("AI feature 1 - Ask the Portfolio (LLM, grounded)", H),
        P("<b>How:</b> question -> LLM (or rule-based parser if no key) -> JSON query spec limited to whitelisted metrics/dimensions/filters -> pandas computes on "
          "the full dataset -> answer + table + source line. The LLM never sees customer rows or does arithmetic; its wording is rejected unless every number appears in the result table. "
          "<b>Trigger:</b> analyst submits a question. <b>False positive:</b> question mapped to the wrong metric/filter (e.g. 'risky' read as default rate) - mitigated by showing the interpreted spec. "
          "<b>False negative:</b> valid question declined because it needs a field outside the whitelist (safe failure; forecasts, profit, PII are declined by design). "
          "<b>Human review:</b> analyst checks interpretation + source table; assistant is read-only."),
        Paragraph("AI feature 2 - Anomaly detection (Isolation Forest + z-scores)", H),
        P("<b>How:</b> each account is compared with its own history (last 3 months vs first 9: DPD z-score, sudden misses, EMI shortfall) and with its peer segment (loan type x income band; "
          "robust z on DPD, missed rate, bureau utilisation/DPD, lines, debt/income). Isolation Forest ranks overall unusualness; top 3% (441 accounts) are flagged with plain-English reasons. "
          "<b>Trigger:</b> score in top 3% (batch, at pipeline build). <b>False positive:</b> a one-off late month or a legitimate high-leverage borrower flagged. "
          "<b>False negative:</b> slow DPD creep or peer-typical but unsafe behaviour is missed (the PD model covers part of this). "
          "<b>Human review:</b> analyst sees reasons and records Confirm / Dismiss + note; saved with reviewer and timestamp, used to tune thresholds. Nothing is auto-actioned."),
        Paragraph("Model results and approach", H),
        P(f"Target: <i>target_default</i>. Features (41): loan terms, demographics, bureau aggregates (write-offs, DPD, utilisation, lines) and 12-month repayment behaviour "
          f"(missed/late rates, DPD level and trend, recent misses, longest miss streak, paydown). Collections data excluded (post-default leakage). 70/30 stratified split, median imputation + missing flags."),
        Table([["Model", "AUC", "KS", "Gini", "5-fold CV AUC / Gini"],
               ["Logistic regression (baseline)", lr["auc"], lr["ks"], lr["gini"], f"{cv['lr']['auc']} / {cv['lr']['gini']}"],
               ["HistGradientBoosting (tuned)", gb["auc"], gb["ks"], gb["gini"], f"{cv['gbm']['auc']} / {cv['gbm']['gini']}"]],
              colWidths=[58 * mm, 20 * mm, 20 * mm, 20 * mm, 50 * mm], style=TableStyle([
                  ("FONTSIZE", (0, 0), (-1, -1), 7), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                  ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)])),
        P("Signal is moderate (synthetic data): the GBM matches the baseline, so I would favour logistic regression for explainability, and use out-of-fold GBM probabilities for scoring. "
          "Top drivers (permutation importance): annual income, years employed, repayment mean DPD, age, loan-to-income, bureau DPD. "
          "<b>Monitoring:</b> monthly PSI on score and inputs (alert >0.10), AUC/KS/Gini and calibration by vintage once outcomes mature, data-quality and feed checks, "
          "age-band parity review, quarterly retrain, anomaly confirm/dismiss ratio."),
        Paragraph("Top 3 portfolio insights", H),
        P("<b>1. Mortgages are the concentration risk:</b> 10% of accounts, 52% of exposure and 65% of defaulted exposure (24.7% default rate) - monitor and tighten mortgage underwriting first. "
          "<b>2. Affordability beats geography:</b> income &lt;40k defaults at 32% vs 9.5% for 100k+, unemployed 35%, while regions only span 15-17%; cross-cut: unemployed on &lt;60k income 3x salaried 100k+, "
          "and under-30 mortgages 32%. <b>3. Missed payments are an early warning:</b> default rate 15% / 23% / 33% with 0 / 1 / 2 misses in the last 3 months - trigger outreach at the first miss."),
        Paragraph("Key assumptions and production needs", H),
        P("<b>Assumptions:</b> target_default is a final outcome for the 12 observed months; repayment months are pre-default observation; bureau data is a snapshot at origination or later "
          "(timing unknown - needs confirming to avoid leakage); missing values are random (documented, flagged, not dropped); bands/thresholds are analyst judgement. "
          "<b>From bank IT/data:</b> governed daily extracts or CDC feeds with snapshot dates for bureau and repayment; a precise default definition (e.g. 90+ DPD) and observation/performance windows; "
          "SSO/RBAC, PII masking and audit trail; a database for review decisions instead of local files; LLM via an approved private endpoint (Azure OpenAI) with data-sharing sign-off; "
          "model-risk/fair-lending review, MLOps (scheduled retrain, monitoring alerts) and a hosted environment inside the bank's network."),
    ]
    doc.build(story)


def main():
    DOCS.mkdir(exist_ok=True)
    architecture()
    solution()
    print("docs written")


if __name__ == "__main__":
    main()
