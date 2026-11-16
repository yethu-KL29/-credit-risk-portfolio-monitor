"""Ingestion: load the 5 raw tables, audit data quality, join to one account-grain dataset.

Join logic (grain = loan account, 14,685 rows):
  loan_accounts (spine)
    LEFT JOIN customers            ON customer_id   (many accounts -> 1 customer)
    LEFT JOIN bureau aggregates    ON customer_id   (bureau lines aggregated to customer first)
    LEFT JOIN repayment aggregates ON account_id    (monthly rows aggregated to account first)
    LEFT JOIN collections aggregates ON account_id  (events aggregated to account first)
Aggregating each child table BEFORE joining prevents row fan-out, so the
account count (and therefore any default-rate / exposure sum) is never inflated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
DATASET_PATH = PROC / "account_dataset.parquet"
DQ_PATH = PROC / "dq_log.json"

OBS_MONTHS = 12  # every account has 12 observed months of repayment history


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(RAW / "customers.csv", parse_dates=["customer_since"]),
        "loans": pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"]),
        "bureau": pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"]),
        "repay": pd.read_csv(RAW / "repayment_history.csv"),
        "coll": pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"]),
    }


def audit(t: dict[str, pd.DataFrame]) -> list[dict]:
    """Record every data-quality issue found and the decision taken."""
    c, l, b, r, e = t["customers"], t["loans"], t["bureau"], t["repay"], t["coll"]
    log: list[dict] = []

    def add(table, issue, count, decision):
        log.append({"table": table, "issue": issue, "rows_affected": int(count), "decision": decision})

    add("customers", "gender coded 'M'/'F' as well as 'Male'/'Female'", c.gender.isin(["M", "F"]).sum(),
        "Standardised to Male/Female.")
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        add("customers", f"missing {col}", c[col].isna().sum(),
            "Kept rows. Median-imputed for the model + '<col>_missing' flag; shown as 'Unknown' in analytics.")
    add("loan_accounts", "missing interest_rate_pct", l.interest_rate_pct.isna().sum(),
        "Kept rows. Imputed with the loan-type median for the model + 'rate_missing' flag.")
    add("bureau_credit", "missing days_past_due", b.days_past_due.isna().sum(),
        "Kept lines. Ignored in max/mean DPD (nan-aware); count of unknown-DPD lines kept as a feature.")
    add("bureau_credit", "date_closed null", b.date_closed.isna().sum(),
        "Expected: null for Active lines only (verified); no action.")
    add("keys", "duplicate customer_id / account_id / bureau_id",
        c.customer_id.duplicated().sum() + l.account_id.duplicated().sum() + b.bureau_id.duplicated().sum(),
        "None found; keys unique.")
    add("keys", "loan accounts with no customer record", (~l.customer_id.isin(c.customer_id)).sum(),
        "None found. LEFT joins from the loan spine used so no account could be dropped silently.")
    add("keys", "repayment rows with no loan account", (~r.account_id.isin(l.account_id)).sum(), "None found.")
    add("keys", "collections rows with no loan account", (~e.account_id.isin(l.account_id)).sum(), "None found.")
    add("repayment_history", "duplicate (account, month) rows", r.duplicated(["account_id", "payment_month"]).sum(),
        "None found.")
    add("repayment_history", "emi_paid > emi_due", (r.emi_paid > r.emi_due).sum(), "None found.")
    add("bureau_credit", "outstanding > sanctioned", (b.outstanding_amount > b.sanctioned_amount).sum(), "None found.")
    add("loan_accounts", "loan_amount outliers (> 99th pct)", (l.loan_amount > l.loan_amount.quantile(.99)).sum(),
        "Kept (mostly mortgages, legitimate). Model uses log(loan_amount) and loan-to-income to tame skew.")
    add("collections_events", "collections activity is a consequence of delinquency",
        e.account_id.nunique(),
        "EXCLUDED from model features (post-event leakage). Used only for dashboard context.")
    add("customers", "customers without loan accounts", (~c.customer_id.isin(l.customer_id)).sum(),
        "Kept in customer table; irrelevant to account-grain dataset.")
    return log


def _bureau_features(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["active"] = (b.account_status == "Active").astype(int)
    b["written_off"] = (b.account_status == "Written-Off").astype(int)
    b["dpd30"] = (b.days_past_due >= 30).astype(int)
    b["dpd_missing"] = b.days_past_due.isna().astype(int)
    b["wo_amount"] = b.outstanding_amount.where(b.written_off == 1, 0)
    g = b.groupby("customer_id").agg(
        bureau_lines=("bureau_id", "count"),
        bureau_active=("active", "sum"),
        bureau_writeoffs=("written_off", "sum"),
        bureau_dpd30_lines=("dpd30", "sum"),
        bureau_dpd_unknown=("dpd_missing", "sum"),
        bureau_sanctioned=("sanctioned_amount", "sum"),
        bureau_outstanding=("outstanding_amount", "sum"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_mean_dpd=("days_past_due", "mean"),
        bureau_first_open=("date_opened", "min"),
    )
    g["bureau_utilisation"] = (g.bureau_outstanding / g.bureau_sanctioned).clip(0, 1.5)
    g["bureau_has_writeoff"] = (g.bureau_writeoffs > 0).astype(int)
    g["bureau_nonbank_share"] = (
        b.assign(nb=(b.lender_category != "Bank").astype(int)).groupby("customer_id").nb.mean()
    )
    return g.reset_index()


def _repay_features(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["account_id", "payment_month"]).copy()
    r["missed"] = (r.payment_status == "Missed").astype(int)
    r["late"] = (r.payment_status == "Late").astype(int)
    r["pay_ratio"] = (r.emi_paid / r.emi_due.replace(0, np.nan)).fillna(1.0)
    last = r.payment_month > r.groupby("account_id").payment_month.transform("max") - 3
    first = r.payment_month <= 9
    r["missed_l3"] = r.missed.where(last, 0)
    r["dpd_l3"] = r.days_past_due.where(last)
    r["dpd_f9"] = r.days_past_due.where(first)

    # longest run of consecutive missed months
    def max_run(s: pd.Series) -> int:
        best = cur = 0
        for v in s.to_numpy():
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    runs = r.groupby("account_id").missed.apply(max_run).rename("rep_max_consec_missed")
    g = r.groupby("account_id").agg(
        rep_months=("payment_month", "count"),
        rep_missed_rate=("missed", "mean"),
        rep_late_rate=("late", "mean"),
        rep_mean_dpd=("days_past_due", "mean"),
        rep_max_dpd=("days_past_due", "max"),
        rep_pay_ratio=("pay_ratio", "mean"),
        rep_missed_l3=("missed_l3", "sum"),
        rep_dpd_l3=("dpd_l3", "mean"),
        rep_dpd_f9=("dpd_f9", "mean"),
        rep_emi=("emi_due", "mean"),
        rep_bal_first=("outstanding_balance", "first"),
        rep_bal_last=("outstanding_balance", "last"),
    )
    g["rep_dpd_trend"] = g.rep_dpd_l3 - g.rep_dpd_f9
    g["rep_balance_paydown"] = 1 - g.rep_bal_last / g.rep_bal_first.replace(0, np.nan)
    return g.join(runs).drop(columns=["rep_bal_first", "rep_bal_last"]).reset_index()


def _coll_features(e: pd.DataFrame) -> pd.DataFrame:
    g = e.groupby("account_id").agg(
        coll_events=("collection_event_id", "count"),
        coll_recovered=("amount_recovered", "sum"),
    )
    g["coll_legal"] = e[e.action_type.isin(["Legal Notice", "Written-Off Referral"])].groupby("account_id").size()
    return g.fillna({"coll_legal": 0}).reset_index()


def build_dataset() -> tuple[pd.DataFrame, list[dict]]:
    t = load_raw()
    dq = audit(t)
    c = t["customers"].copy()
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})

    df = t["loans"].merge(c, on="customer_id", how="left", validate="m:1")
    df = df.merge(_bureau_features(t["bureau"]), on="customer_id", how="left", validate="m:1")
    df = df.merge(_repay_features(t["repay"]), on="account_id", how="left", validate="1:1")
    df = df.merge(_coll_features(t["coll"]), on="account_id", how="left", validate="1:1")

    # customers without bureau lines / accounts without collections: structural zeros, not missing
    zero_cols = [c_ for c_ in df.columns if c_.startswith("coll_")] + [
        "bureau_lines", "bureau_active", "bureau_writeoffs", "bureau_dpd30_lines", "bureau_dpd_unknown",
        "bureau_sanctioned", "bureau_outstanding", "bureau_has_writeoff", "bureau_utilisation",
        "bureau_nonbank_share",
    ]
    no_bureau = int((df.bureau_lines.isna()).sum())
    df[zero_cols] = df[zero_cols].fillna(0)
    dq.append({"table": "bureau_credit", "issue": "accounts whose customer has no bureau line", "rows_affected": no_bureau,
               "decision": "Bureau counts/amounts set to 0 (genuinely no external credit); DPD left null."})
    dq.append({"table": "collections_events", "issue": "accounts with no collections event",
               "rows_affected": int((df.coll_events == 0).sum()), "decision": "Event counts set to 0 (structural zero)."})

    # analysis-friendly bands (used by analytics, dashboard and Ask-the-Portfolio)
    df["age_band"] = pd.cut(df.age, [0, 29, 39, 49, 59, 120], labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    df["income_band"] = pd.cut(df.annual_income, [0, 40000, 60000, 80000, 100000, 1e9],
                               labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"]).astype(str)
    df["origination_year"] = df.origination_date.dt.year
    df["collateral"] = np.where(df.collateral_flag == 1, "Secured", "Unsecured")
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        df[col + "_missing"] = df[col].isna().astype(int)
    df["rate_missing"] = df.interest_rate_pct.isna().astype(int)
    df["loan_to_income"] = df.loan_amount / df.annual_income
    return df, dq


def main() -> pd.DataFrame:
    PROC.mkdir(parents=True, exist_ok=True)
    df, dq = build_dataset()
    assert df.account_id.is_unique and len(df) == 14685, "join changed account grain"
    df.to_parquet(DATASET_PATH, index=False)
    DQ_PATH.write_text(json.dumps(dq, indent=2))
    print(f"ingest: {len(df):,} accounts x {df.shape[1]} columns -> {DATASET_PATH.name}")
    return df


if __name__ == "__main__":
    main()
