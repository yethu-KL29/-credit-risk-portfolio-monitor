"""Default model: logistic-regression baseline + HistGradientBoosting, scoring and per-account drivers."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ingest import DATASET_PATH, PROC

SEED = 42
MODEL_PATH = PROC / "model.joblib"
SCORES_PATH = PROC / "scores.parquet"
METRICS_PATH = PROC / "model_metrics.json"

CATEGORICAL = ["loan_type", "employment_type", "housing_type", "disbursal_channel", "education_level",
               "marital_status", "region", "city_tier"]
NUMERIC = [
    # customer / loan
    "age", "annual_income", "years_employed", "num_dependents", "years_at_residence", "loan_amount",
    "interest_rate_pct", "tenure_months", "collateral_flag", "loan_to_income", "rate_missing",
    # bureau (customer-level)
    "bureau_lines", "bureau_active", "bureau_writeoffs", "bureau_has_writeoff", "bureau_dpd30_lines",
    "bureau_dpd_unknown", "bureau_sanctioned", "bureau_outstanding", "bureau_utilisation", "bureau_max_dpd",
    "bureau_mean_dpd", "bureau_nonbank_share",
    # repayment (account-level, first 12 months)
    "rep_missed_rate", "rep_late_rate", "rep_mean_dpd", "rep_max_dpd", "rep_pay_ratio", "rep_missed_l3",
    "rep_dpd_l3", "rep_dpd_trend", "rep_max_consec_missed", "rep_balance_paydown",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "target_default"

FEATURE_GROUPS = {  # used for per-account driver explanations
    "Income / affordability": ["annual_income", "loan_to_income", "loan_amount"],
    "Employment": ["employment_type", "years_employed"],
    "Age": ["age"],
    "Loan terms / pricing": ["loan_type", "interest_rate_pct", "tenure_months", "collateral_flag", "rate_missing",
                             "disbursal_channel"],
    "Bureau: write-offs / delinquency": ["bureau_writeoffs", "bureau_has_writeoff", "bureau_dpd30_lines",
                                         "bureau_max_dpd", "bureau_mean_dpd", "bureau_dpd_unknown"],
    "Bureau: leverage / depth": ["bureau_lines", "bureau_active", "bureau_sanctioned", "bureau_outstanding",
                                 "bureau_utilisation", "bureau_nonbank_share"],
    "Repayment: missed / late behaviour": ["rep_missed_rate", "rep_late_rate", "rep_missed_l3",
                                           "rep_max_consec_missed", "rep_pay_ratio"],
    "Repayment: days past due": ["rep_mean_dpd", "rep_max_dpd", "rep_dpd_l3", "rep_dpd_trend"],
    "Repayment: balance paydown": ["rep_balance_paydown"],
    "Stability / demographics": ["housing_type", "years_at_residence", "num_dependents", "marital_status",
                                 "education_level", "region", "city_tier"],
}


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(ks_stat(y, p), 4)}


def _prep(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])


def make_lr() -> Pipeline:
    return Pipeline([("prep", _prep(True)), ("clf", LogisticRegression(C=0.5, max_iter=2000))])


def make_gbm() -> Pipeline:
    return Pipeline([("prep", _prep(False)), ("clf", HistGradientBoostingClassifier(
        learning_rate=0.02, max_iter=300, max_leaf_nodes=3, min_samples_leaf=150, l2_regularization=5.0,
        early_stopping=False, random_state=SEED))])


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drivers(model: Pipeline, X: pd.DataFrame, base: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Occlusion drivers: how much does the score change when a feature group is replaced by the portfolio
    median/mode? Positive = pushes risk up for this account."""
    p0 = model.predict_proba(X)[:, 1]
    deltas = {}
    for name, cols in FEATURE_GROUPS.items():
        Xc = X.copy()
        for c in cols:
            Xc[c] = base[c]
        deltas[name] = p0 - model.predict_proba(Xc)[:, 1]
    D = pd.DataFrame(deltas, index=X.index)
    return D


def main() -> dict:
    df = pd.read_parquet(DATASET_PATH)
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    lr, gbm = make_lr().fit(Xtr, ytr), make_gbm().fit(Xtr, ytr)
    res = {"lr": metrics(yte, lr.predict_proba(Xte)[:, 1]), "gbm": metrics(yte, gbm.predict_proba(Xte)[:, 1]),
           "n_train": len(Xtr), "n_test": len(Xte), "default_rate": round(float(y.mean()), 4)}
    # 5-fold CV on the whole set to show the estimate is stable
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_gbm = cross_val_predict(make_gbm(), X, y, cv=cv, method="predict_proba")[:, 1]
    oof_lr = cross_val_predict(make_lr(), X, y, cv=cv, method="predict_proba")[:, 1]
    res["cv_oof"] = {"lr": metrics(y, oof_lr), "gbm": metrics(y, oof_gbm)}

    # feature importance (permutation on hold-out, AUC drop) for the GBM
    pi = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop": pi.importances_mean}).sort_values("auc_drop", ascending=False)
    res["importance"] = imp.head(15).round(5).to_dict("records")

    # LR coefficients (transparent baseline)
    names = lr.named_steps["prep"].get_feature_names_out()
    coef = pd.DataFrame({"feature": names, "coef": lr.named_steps["clf"].coef_[0]})
    coef["abs"] = coef.coef.abs()
    res["lr_coefficients"] = coef.sort_values("abs", ascending=False).head(15).drop(columns="abs").round(4).to_dict("records")

    # calibration by decile on hold-out
    pte = gbm.predict_proba(Xte)[:, 1]
    cal = pd.DataFrame({"p": pte, "y": yte.values})
    cal["decile"] = pd.qcut(cal.p, 10, labels=False, duplicates="drop") + 1
    res["calibration"] = cal.groupby("decile").agg(avg_pd=("p", "mean"), actual=("y", "mean")).round(4).reset_index().to_dict("records")

    # final model: refit on all data for scoring; honest OOF score kept alongside
    final = make_gbm().fit(X, y)
    base = X.copy()
    for c in NUMERIC:
        base[c] = X[c].median()
    for c in CATEGORICAL:
        base[c] = X[c].mode().iloc[0]
    D = drivers(final, X, base)

    out = df[["account_id", "customer_id"]].copy()
    out["pd_score"] = oof_gbm  # out-of-fold: account was never scored by a model that saw its label
    out["pd_lr"] = oof_lr
    out["risk_band"] = pd.cut(out.pd_score, [0, .10, .20, .35, 1.0], labels=["Low", "Medium", "High", "Very High"]).astype(str)
    order = np.argsort(-D.to_numpy(), axis=1)
    cols = np.array(D.columns)
    for k in range(3):
        out[f"driver{k+1}"] = cols[order[:, k]]
        out[f"driver{k+1}_delta"] = np.take_along_axis(D.to_numpy(), order[:, [k]], axis=1)[:, 0].round(4)
    out.to_parquet(SCORES_PATH, index=False)
    D.assign(account_id=df.account_id.values).to_parquet(PROC / "driver_deltas.parquet", index=False)

    # monitoring reference: PSI of score + key features, early vs late origination cohort
    early = df.origination_year <= 2020
    res["monitoring"] = {
        "score_psi_early_vs_late": round(psi(oof_gbm[early.values], oof_gbm[~early.values]), 4),
        "feature_psi": {f: round(psi(df.loc[early, f].fillna(df[f].median()).values,
                                     df.loc[~early, f].fillna(df[f].median()).values), 4)
                        for f in ["annual_income", "rep_missed_rate", "rep_mean_dpd", "bureau_utilisation", "loan_amount"]},
        "cohort_definition": "origination 2018-2020 vs 2021-2023",
    }
    joblib.dump({"model": final, "base": base.iloc[:1]}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(res, indent=2))
    print("model:", json.dumps({k: res[k] for k in ["lr", "gbm", "cv_oof"]}))
    return res


if __name__ == "__main__":
    main()
