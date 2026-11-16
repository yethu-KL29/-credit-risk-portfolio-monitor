"""Anomaly detection: unusual repayment / bureau behaviour vs the account's OWN history and vs its PEER segment.

Method (explainable by design):
  1. Own-history signal: last-3-month DPD vs the account's first-9-month mean/std (z-score) + "sudden stop" (>=2
     missed payments in last 3 months after at most 1 in the first 9) + EMI shortfall in last 3 months.
  2. Peer signal: robust z-score (median/MAD) of repayment and bureau features within the loan-type segment x
     income band (so a mortgage is compared to mortgages, not credit cards).
  3. IsolationForest over all z-signals ranks "overall unusualness"; the top ANOMALY_RATE are flagged and each
     flag carries human-readable reasons derived from the z-signals that exceeded their threshold.
Every flag is a review candidate, never an automatic action - see the human-in-the-loop decision log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .ingest import DATASET_PATH, PROC, RAW

FLAGS_PATH = PROC / "anomaly_flags.parquet"
ANOMALY_RATE = 0.03
PEER_FEATURES = {
    "rep_mean_dpd": "average days-past-due",
    "rep_missed_rate": "missed-payment rate",
    "rep_max_dpd": "worst days-past-due",
    "bureau_utilisation": "bureau credit utilisation",
    "bureau_max_dpd": "worst bureau days-past-due",
    "bureau_lines": "number of bureau credit lines",
    "bureau_leverage": "external debt / income",
}
PEER_Z_THRESHOLD = 3.0
OWN_Z_THRESHOLD = 2.5


def _robust_z(s: pd.Series, grp: pd.Series) -> pd.Series:
    med = s.groupby(grp).transform("median")
    mad = (s - med).abs().groupby(grp).transform("median") * 1.4826
    sd = s.groupby(grp).transform("std")
    scale = mad.where(mad > 1e-9, sd).replace(0, np.nan)  # MAD collapses on zero-inflated features
    return ((s - med) / scale).fillna(0).clip(-10, 10)


def _own_history() -> pd.DataFrame:
    r = pd.read_csv(RAW / "repayment_history.csv").sort_values(["account_id", "payment_month"])
    r["missed"] = (r.payment_status == "Missed").astype(int)
    r["short"] = (1 - r.emi_paid / r.emi_due.replace(0, np.nan)).fillna(0).clip(0, 1)
    early, late = r[r.payment_month <= 9], r[r.payment_month >= 10]
    e = early.groupby("account_id").agg(f9_mean=("days_past_due", "mean"), f9_std=("days_past_due", "std"),
                                         f9_missed=("missed", "sum"))
    l = late.groupby("account_id").agg(l3_mean=("days_past_due", "mean"), l3_missed=("missed", "sum"),
                                        l3_short=("short", "mean"))
    g = e.join(l)
    g["own_dpd_z"] = ((g.l3_mean - g.f9_mean) / g.f9_std.fillna(0).clip(lower=3)).fillna(0)
    g["own_sudden_stop"] = ((g.l3_missed >= 2) & (g.f9_missed <= 1)).astype(int)
    return g.reset_index()


def main() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df = df.merge(_own_history(), on="account_id", how="left")
    df["bureau_leverage"] = df.bureau_outstanding / df.annual_income
    peer = df.loan_type + " | " + df.income_band
    Z = pd.DataFrame({f"z_{f}": _robust_z(df[f].fillna(df[f].median()), peer) for f in PEER_FEATURES})
    Z["own_dpd_z"] = df.own_dpd_z.clip(-10, 10)
    Z["own_sudden_stop"] = df.own_sudden_stop * 5.0
    Z["own_short"] = (df.l3_short.fillna(0) * 10).clip(0, 10)

    iso = IsolationForest(n_estimators=300, contamination=ANOMALY_RATE, random_state=42).fit(Z)
    df["anomaly_score"] = -iso.score_samples(Z)  # higher = more unusual
    thr = df.anomaly_score.quantile(1 - ANOMALY_RATE)

    def reasons(i: int) -> list[str]:
        out = []
        for f, label in PEER_FEATURES.items():
            z = Z.at[i, f"z_{f}"]
            if z >= PEER_Z_THRESHOLD:
                out.append(f"Peer: {label} is {df.at[i, f]:,.2f} vs peer median "
                           f"{df.loc[peer == peer[i], f].median():,.2f} (z={z:.1f}) in {peer[i]}")
        if df.at[i, "own_dpd_z"] >= OWN_Z_THRESHOLD:
            out.append(f"Own history: avg DPD last 3 months {df.at[i, 'l3_mean']:.0f} vs {df.at[i, 'f9_mean']:.0f} "
                       f"over the first 9 months (z={df.at[i, 'own_dpd_z']:.1f})")
        if df.at[i, "own_sudden_stop"]:
            out.append(f"Own history: {int(df.at[i, 'l3_missed'])} missed payments in last 3 months after "
                       f"{int(df.at[i, 'f9_missed'])} in the first 9")
        if df.at[i, "l3_short"] >= 0.5:
            out.append(f"Own history: paid only {1 - df.at[i, 'l3_short']:.0%} of EMI in last 3 months")
        return out

    flagged_idx = df.index[df.anomaly_score >= thr]
    rows = []
    for i in flagged_idx:
        rs = reasons(i)
        if not rs:  # multivariate outlier with no single extreme feature
            rs = ["Multivariate: unusual combination of repayment and bureau signals (no single extreme feature)"]
        kind = ("Own history" if any(r.startswith("Own") for r in rs) else "") 
        kind += (" + " if kind and any(r.startswith("Peer") for r in rs) else "")
        kind += ("Peer segment" if any(r.startswith("Peer") for r in rs) else "")
        rows.append({"account_id": df.at[i, "account_id"], "customer_id": df.at[i, "customer_id"],
                     "anomaly_score": round(float(df.at[i, "anomaly_score"]), 4),
                     "flag_type": kind or "Multivariate", "n_reasons": len(rs), "reasons": " || ".join(rs),
                     "peer_group": peer[i]})
    out = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    out.to_parquet(FLAGS_PATH, index=False)
    df[["account_id", "anomaly_score"]].to_parquet(PROC / "anomaly_scores_all.parquet", index=False)
    print(f"anomaly: flagged {len(out):,} of {len(df):,} accounts ({out.flag_type.value_counts().to_dict()})")
    return out


if __name__ == "__main__":
    main()
