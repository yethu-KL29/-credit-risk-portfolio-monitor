"""Ask the Portfolio: grounded natural-language Q&A.

Design - the LLM never sees row-level data and never produces numbers on its own:
  question --(LLM or rule-based parser)--> validated QUERY SPEC (whitelisted metric / dimensions / filters)
           --(pandas)--> result table  --> answer text + source line
  * Numbers shown to the analyst always come from the pandas result table.
  * If an LLM is configured it may phrase the answer, but the phrasing is accepted only if every number it
    contains appears in the result table; otherwise the deterministic template answer is used.
  * Questions outside the whitelist (forecasts, profit, PII, unknown fields...) are declined.
Without an API key the same pipeline runs with the built-in rule-based parser.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from . import llm

DIMENSIONS = {
    "loan_type": "loan type", "region": "region", "employment_type": "employment type", "age_band": "age band",
    "income_band": "income band", "city_tier": "city tier", "housing_type": "housing type",
    "disbursal_channel": "disbursal channel", "gender": "gender", "collateral": "secured/unsecured",
    "origination_year": "origination year", "risk_band": "model risk band", "marital_status": "marital status",
    "education_level": "education level",
}
METRICS = {
    "default_rate": "default rate", "accounts": "number of accounts", "exposure": "total exposure (sum of loan amount, USD)",
    "defaulted_exposure": "defaulted exposure (loan amount of defaulted accounts, USD)",
    "avg_loan_amount": "average loan amount (USD)", "avg_income": "average annual income of account holders (USD)",
    "avg_pd": "average model-predicted default probability", "avg_interest_rate": "average interest rate (%)",
    "anomaly_flags": "number of accounts flagged by anomaly detection",
}
PCT_METRICS = {"default_rate", "avg_pd"}
MONEY_METRICS = {"exposure", "defaulted_exposure", "avg_loan_amount", "avg_income"}
MIN_GROUP = 30
DECLINE_TERMS = [
    "forecast", "predict next", "next year", "next quarter", "next month", "in 2025", "in 2026", "in 2027", "future",
    "profit", "revenue", "income statement", "balance sheet", "share price", "stock", "weather", "ceo", "address",
    "phone", "email", "ssn", "passport", "customer name", "who is", "cibil", "fico", "credit score", "npa ratio",
    "recovery rate", "lgd", "ead", "provision", "competitor", "market share", "nationality", "religion", "race",
]


@dataclass
class Result:
    ok: bool
    answer: str
    table: pd.DataFrame | None = None
    source: str = ""
    spec: dict = field(default_factory=dict)
    mode: str = "rules"


def prepare(df: pd.DataFrame, scores: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(scores[["account_id", "pd_score", "risk_band"]], on="account_id", how="left")
    d["anomaly_flag"] = d.account_id.isin(flags.account_id).astype(int)
    d["origination_year"] = d.origination_year.astype(str)
    return d


def _metric_value(g: pd.DataFrame, metric: str) -> float:
    if metric == "default_rate":
        return g.target_default.mean()
    if metric == "accounts":
        return float(len(g))
    if metric == "exposure":
        return g.loan_amount.sum()
    if metric == "defaulted_exposure":
        return g.loc[g.target_default == 1, "loan_amount"].sum()
    if metric == "avg_loan_amount":
        return g.loan_amount.mean()
    if metric == "avg_income":
        return g.annual_income.mean()
    if metric == "avg_pd":
        return g.pd_score.mean()
    if metric == "avg_interest_rate":
        return g.interest_rate_pct.mean()
    if metric == "anomaly_flags":
        return float(g.anomaly_flag.sum())
    raise ValueError(metric)


def validate_spec(spec: dict, data: pd.DataFrame) -> dict:
    """Whitelist validation - anything unknown raises ValueError."""
    if spec.get("decline"):
        return spec
    m = spec.get("metric")
    if m not in METRICS:
        raise ValueError(f"unknown metric {m!r}")
    gb = spec.get("group_by") or []
    if isinstance(gb, str):
        gb = [gb]
    if len(gb) > 2 or any(g not in DIMENSIONS for g in gb):
        raise ValueError("invalid group_by")
    filters = spec.get("filters") or {}
    clean = {}
    for k, v in filters.items():
        if k not in DIMENSIONS:
            raise ValueError(f"invalid filter field {k!r}")
        vals = [v] if isinstance(v, (str, int)) else list(v)
        vals = [str(x) for x in vals]
        valid = set(data[k].astype(str).unique())
        bad = [x for x in vals if x not in valid]
        if bad:
            raise ValueError(f"value {bad} not found in {k}")
        clean[k] = vals
    sort = spec.get("sort", "desc")
    limit = spec.get("limit")
    return {"metric": m, "group_by": gb, "filters": clean, "sort": sort if sort in ("asc", "desc") else "desc",
            "limit": int(limit) if limit else None}


def execute(spec: dict, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    d = data
    for k, vals in spec["filters"].items():
        d = d[d[k].astype(str).isin(vals)]
    metric, gb = spec["metric"], spec["group_by"]
    label = METRICS[metric]
    if d.empty:
        return pd.DataFrame(), "no accounts match the filters"
    if gb:
        rows = [{**dict(zip(gb, key if isinstance(key, tuple) else (key,))), "accounts": len(g),
                 "defaults": int(g.target_default.sum()), label: _metric_value(g, metric)}
                for key, g in d.groupby(gb, observed=True)]
        t = pd.DataFrame(rows)
        t["small_sample"] = t.accounts < MIN_GROUP
        t = t.sort_values(label, ascending=spec["sort"] == "asc").reset_index(drop=True)
    else:
        t = pd.DataFrame([{"accounts": len(d), "defaults": int(d.target_default.sum()), label: _metric_value(d, metric)}])
        t["small_sample"] = len(d) < MIN_GROUP
    f = "; ".join(f"{DIMENSIONS[k]} in {v}" for k, v in spec["filters"].items()) or "none"
    src = (f"account_dataset.parquet (14,685 loan accounts, joined from the 5 source tables) | metric: {label} | "
           f"grouped by: {', '.join(gb) or '-'} | filters: {f} | accounts in scope: {len(d):,}")
    return t, src


def fmt(metric: str, v: float) -> str:
    if metric in PCT_METRICS:
        return f"{v:.1%}"
    if metric in MONEY_METRICS:
        return f"${v:,.0f}"
    if metric == "avg_interest_rate":
        return f"{v:.2f}%"
    return f"{v:,.0f}"


def template_answer(spec: dict, t: pd.DataFrame) -> str:
    metric, gb = spec["metric"], spec["group_by"]
    label = METRICS[metric]
    if t.empty:
        return "No accounts match those filters."
    if not gb:
        r = t.iloc[0]
        if metric == "accounts":
            return f"There are {int(r.accounts):,} accounts in scope."
        return f"The {label} is {fmt(metric, r[label])} across {int(r.accounts):,} accounts."
    top = t.iloc[0]
    name = lambda r: " / ".join(str(r[g]) for g in gb)  # noqa: E731
    word = "highest" if spec["sort"] == "desc" else "lowest"
    s = f"The {word} {label} is {name(top)} at {fmt(metric, top[label])} ({int(top.accounts):,} accounts)."
    if len(t) > 1:
        bot = t.iloc[-1]
        s += f" The {'lowest' if word == 'highest' else 'highest'} is {name(bot)} at {fmt(metric, bot[label])}."
    if t.small_sample.any():
        s += f" Note: groups with fewer than {MIN_GROUP} accounts are marked as small samples."
    return s


# ---------------------------------------------------------------- rule-based parser (no API key needed)
_DIM_SYN = {
    "loan_type": ["loan type", "product", "loan types", "products"], "region": ["region", "regions"],
    "employment_type": ["employment", "employment type", "job"], "age_band": ["age", "age band", "age group"],
    "income_band": ["income", "income band", "income group"], "city_tier": ["city tier", "tier"],
    "housing_type": ["housing"], "disbursal_channel": ["channel", "disbursal"], "gender": ["gender", "sex"],
    "collateral": ["collateral", "secured", "unsecured"], "origination_year": ["year", "vintage", "cohort"],
    "risk_band": ["risk band", "risk bands", "risk segment"], "marital_status": ["marital"],
    "education_level": ["education"],
}
_METRIC_SYN = [
    ("defaulted_exposure", ["defaulted exposure", "exposure at default", "amount defaulted", "defaulted amount"]),
    ("default_rate", ["default rate", "defaulted", "default", "riskiest", "risky", "delinquen", "bad rate"]),
    ("anomaly_flags", ["anomal", "flagged", "unusual"]),
    ("avg_pd", ["predicted", "model score", "avg pd", "probability", "pd score"]),
    ("exposure", ["exposure", "total loan", "portfolio size", "outstanding", "disbursed", "total amount"]),
    ("avg_loan_amount", ["average loan", "avg loan", "mean loan", "ticket size", "loan size"]),
    ("avg_income", ["average income", "avg income", "mean income"]),
    ("avg_interest_rate", ["interest rate", "pricing", "rate"]),
    ("accounts", ["how many", "number of", "count", "accounts", "loans"]),
]


def rule_parse(q: str, data: pd.DataFrame) -> dict:
    ql = q.lower()
    for term in DECLINE_TERMS:
        if term in ql:
            return {"decline": f"The dashboard data cannot answer questions involving '{term}'."}
    metric = next((m for m, kws in _METRIC_SYN if any(k in ql for k in kws)), None)
    filters: dict[str, list[str]] = {}
    for dim in DIMENSIONS:
        if dim == "origination_year":
            ys = re.findall(r"\b(20[12]\d)\b", ql)
            if ys:
                filters[dim] = ys
            continue
        for v in data[dim].astype(str).unique():
            if len(v) > 2 and re.search(rf"(?<![a-z0-9]){re.escape(v.lower())}s?(?![a-z0-9])", ql):
                filters.setdefault(dim, []).append(v)
    # "default"/"defaulted" inside "defaulted exposure" handled by ordering above
    gb = []
    m_by = re.search(r"\b(?:by|per|across|for each|which|what)\b(.*)", ql)
    scope = m_by.group(1) if m_by else ql
    for dim, syns in _DIM_SYN.items():
        if dim in filters and not re.search(r"\bby\b|\bper\b|across|each", ql):
            continue
        if any(re.search(rf"\b{re.escape(s)}\b", scope) for s in syns):
            gb.append(dim)
    gb = gb[:2]
    if metric is None and gb:
        metric = "default_rate"
    if metric is None:
        return {"decline": "I could not map that question to a metric in the portfolio data "
                           "(default rate, accounts, exposure, loan amount, income, interest rate, model PD, anomaly flags)."}
    if not gb and not filters and not re.search(r"overall|portfolio|total|how many|average|number|what is the", ql):
        return {"decline": "Please specify a dimension (e.g. loan type, region) or filter for that metric."}
    # drop dims that are already pinned by filter unless explicit
    sort = "asc" if re.search(r"lowest|least|smallest|minimum|best|safest", ql) else "desc"
    return {"metric": metric, "group_by": gb, "filters": filters, "sort": sort}


# ---------------------------------------------------------------- LLM path
def _schema_prompt(data: pd.DataFrame) -> str:
    vals = {k: sorted(data[k].astype(str).unique())[:12] for k in DIMENSIONS}
    return (
        "You translate analyst questions about a bank loan portfolio into a JSON query spec. You do NOT answer "
        "questions and you never invent numbers. Output ONLY JSON.\n"
        f"metric must be one of: {json.dumps(METRICS)}\n"
        f"group_by: list of up to 2 of: {json.dumps(DIMENSIONS)}\n"
        f"filters: object mapping a dimension to a value or list of values. Allowed values: {json.dumps(vals)}\n"
        'sort: "desc" or "asc" (use asc for lowest/least). \n'
        'If the question cannot be answered from these fields (forecasts, profit, PII, unknown fields, opinions, '
        'causal claims) return {"decline": "<short reason>"}.\n'
        'Example: {"metric":"default_rate","group_by":["loan_type"],"filters":{"region":["North"]},"sort":"desc"}'
    )


def llm_parse(q: str, data: pd.DataFrame) -> dict:
    text = llm.chat([{"role": "system", "content": _schema_prompt(data)}, {"role": "user", "content": q}], json_mode=True)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()  # some models wrap JSON in fences
    return json.loads(text)


def _numbers_ok(text: str, t: pd.DataFrame, extra: str) -> bool:
    allowed = set()
    for v in t.to_numpy().ravel():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            for x in (v, v * 100):
                for dec in (0, 1, 2, 3):
                    allowed.add(round(float(x), dec))
    for tok in re.findall(r"\d[\d,]*\.?\d*", text):
        n = float(tok.replace(",", "").rstrip("."))
        if round(n, 2) in allowed or round(n, 1) in allowed or round(n) in allowed:
            continue
        if tok in extra:
            continue
        return False
    return True


def llm_phrase(q: str, spec: dict, t: pd.DataFrame, fallback: str, extra: str) -> str:
    try:
        shown = t.head(10).copy()
        for c in shown.columns:
            if shown[c].dtype.kind == "f":
                shown[c] = shown[c].round(4)
        text = llm.chat([
            {"role": "system", "content": "Answer the analyst's question in at most 3 short plain-English sentences "
             "using ONLY the numbers in the result table (default_rate/avg_pd are fractions: show as %). "
             "Do not add numbers, causes or recommendations that are not in the table."},
            {"role": "user", "content": f"Question: {q}\nResult table (JSON): {shown.to_json(orient='records')}"}]).strip()
        return text if _numbers_ok(text, t, extra) else fallback
    except Exception:
        return fallback


def ask(question: str, data: pd.DataFrame) -> Result:
    q = (question or "").strip()
    if len(q) < 4:
        return Result(False, "Please type a question about the portfolio.")
    mode, note = "rules", ""
    try:
        if llm.llm_available():
            mode = "llm"
            try:
                spec = llm_parse(q, data)
            except Exception as e:  # network / quota / bad JSON -> degrade gracefully
                mode, note, spec = "rules", f" (LLM unavailable: {type(e).__name__}; used rule-based parser)", rule_parse(q, data)
        else:
            spec = rule_parse(q, data)
        low = q.lower()
        if not spec.get("decline") and any(t in low for t in DECLINE_TERMS):
            spec = {"decline": "That topic is not covered by the dashboard data."}
        spec = validate_spec(spec, data)
    except ValueError as e:
        return Result(False, f"Declined: I can't answer that from the portfolio data ({e}).", mode=mode)
    if spec.get("decline"):
        return Result(False, f"Declined: {spec['decline']} I only answer from the loan, customer, bureau, repayment and "
                             f"collections data in this dashboard.", mode=mode)
    t, src = execute(spec, data)
    answer = template_answer(spec, t)
    if mode == "llm" and not t.empty:
        answer = llm_phrase(q, spec, t, answer, q)
    return Result(True, answer + note, t, src, spec, mode)
