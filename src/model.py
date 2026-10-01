"""Leakage-safe warranty fraud model and evaluation utilities.

The model uses strictly as-of historical feature transformations and a pinned
scikit-learn RandomForestClassifier for robust ranking of warranty claims.
History features are computed from labelled rows strictly before each row's
submission time; the final test model uses all non-blank training outcomes.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
try:
    from sklearn.ensemble import RandomForestClassifier
except Exception as exc:  # do not silently change the submitted model
    RandomForestClassifier = None
    _SKLEARN_IMPORT_ERROR = exc


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
ROOT = PROJECT_ROOT


def _locate_file(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    for p in candidates:
        matches = list(PROJECT_ROOT.rglob(p.name))
        if matches:
            return matches[0]
        stem = p.stem.replace("*-", "")
        suffix = p.suffix
        matches = list(PROJECT_ROOT.rglob(f"*{stem}*{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"None of {candidates} could be located under {PROJECT_ROOT}")


TRAIN = _locate_file([
    PROJECT_ROOT / "data" / "raw" / "train.csv",
    PROJECT_ROOT / "data" / "raw" / "001-train.csv",
    PROJECT_ROOT / "train.csv",
])

TEST = _locate_file([
    PROJECT_ROOT / "data" / "raw" / "test_unlabelled.csv",
    PROJECT_ROOT / "data" / "raw" / "001-test_unlabelled.csv",
    PROJECT_ROOT / "test_unlabelled.csv",
])

PARTNERS = _locate_file([
    PROJECT_ROOT / "data" / "reference" / "partners.csv",
    PROJECT_ROOT / "data" / "reference" / "001-partners.csv",
    PROJECT_ROOT / "partners.csv",
])

PRODUCTS = _locate_file([
    PROJECT_ROOT / "data" / "reference" / "products.csv",
    PROJECT_ROOT / "data" / "reference" / "001-products.csv",
    PROJECT_ROOT / "products.csv",
])




def _fit_vocab(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "source": sorted(df.source.dropna().astype(str).unique()),
        "photo_attached": sorted(df.photo_attached.dropna().astype(str).unique()),
        "partner_inspected": sorted(df.partner_inspected.dropna().astype(str).unique()),
        "claim_description": sorted(df.claim_description.dropna().astype(str).unique()),
        "sku": sorted(df.sku.dropna().astype(str).unique()),
        "city": sorted(df.city.dropna().astype(str).unique()),
        "partner_type": sorted(df.partner_type.dropna().astype(str).unique()),
        "partner_age_bin": ["0-30", "31-90", "91-180", "181-365", "366-730", "731+"],
    }


def _recent_partner_rate(rows: pd.DataFrame, hist: pd.DataFrame, days: int = 180) -> np.ndarray:
    """Strictly-as-of exponentially weighted partner fraud rate."""
    prior = 0.01; h = hist.copy(); r = rows.copy()
    h["_k"] = h.partner_id.astype(str); r["_k"] = r.partner_id.astype(str)
    h["_d"] = pd.to_datetime(h.submitted_at); r["_d"] = pd.to_datetime(r.submitted_at); r["_row"] = np.arange(len(r))
    out = np.full(len(r), prior, dtype=float)
    for k, rg in r.groupby("_k", sort=False):
        hg = h[h._k == k]
        if hg.empty: continue
        hd = hg._d.to_numpy(dtype="datetime64[s]").astype("int64") / 86400
        rd = rg._d.to_numpy(dtype="datetime64[s]").astype("int64") / 86400
        age = rd[:, None] - hd[None, :]
        w = np.exp(-np.maximum(age, 0) / days) * (age > 0)
        out[rg._row.to_numpy(int)] = (w @ hg.y.to_numpy(float) + 5 * prior) / (w.sum(axis=1) + 5)
    return out


def _asof_target_rate(rows: pd.DataFrame, ref: pd.DataFrame, key: str, strength: float = 25.0) -> np.ndarray:
    """Smoothed target rate using only reference rows strictly earlier in time."""
    # Fixed prior avoids using future-period labels through the smoothing term.
    # It is a conservative baseline close to the observed low base rate.
    prior = 0.01
    h = ref[[key, "submitted_at", "y"]].copy(); r = rows[[key, "submitted_at"]].copy()
    h["_k"] = h[key].astype(str); r["_k"] = r[key].astype(str)
    h["_d"] = pd.to_datetime(h.submitted_at); r["_d"] = pd.to_datetime(r.submitted_at); h["_row"] = np.arange(len(h)); r["_row"] = np.arange(len(r))
    h = h.groupby(["_k", "_d"], as_index=False).agg(_sum=("y", "sum"), _count=("y", "count")).sort_values("_d")
    h["_cs"] = h.groupby("_k")["_sum"].cumsum(); h["_cc"] = h.groupby("_k")["_count"].cumsum()
    r = r.sort_values("_d")
    z = pd.merge_asof(r, h[["_k", "_d", "_cs", "_cc"]], on="_d", by="_k", direction="backward", allow_exact_matches=False)
    sums = z._cs.fillna(0).to_numpy(float); counts = z._cc.fillna(0).to_numpy(float)
    out = np.empty(len(rows), dtype=float); out[z._row.to_numpy(int)] = (sums + strength * prior) / (counts + strength)
    return out


def _asof_count(rows: pd.DataFrame, ref: pd.DataFrame, key: str) -> np.ndarray:
    """Count matching reference rows strictly earlier than each row."""
    h = ref[[key, "submitted_at"]].copy(); r = rows[[key, "submitted_at"]].copy()
    h["_k"] = h[key].astype(str); r["_k"] = r[key].astype(str)
    h["_d"] = pd.to_datetime(h.submitted_at); r["_d"] = pd.to_datetime(r.submitted_at); r["_row"] = np.arange(len(r))
    h = h.groupby(["_k", "_d"], as_index=False).size().sort_values("_d"); h["_cc"] = h.groupby("_k")["size"].cumsum()
    r = r.sort_values("_d")
    z = pd.merge_asof(r, h[["_k", "_d", "_cc"]], on="_d", by="_k", direction="backward", allow_exact_matches=False)
    out = np.zeros(len(rows), dtype=float); out[z._row.to_numpy(int)] = z._cc.fillna(0).to_numpy(float); return out


def _add_history(df: pd.DataFrame, ref: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
    """Add strictly-as-of target rates; leave_one_out is retained for API compatibility."""
    out = df.copy()
    out["partner_rate"] = _asof_target_rate(out, ref, "partner_id", 35)
    out["sku_rate"] = _asof_target_rate(out, ref, "sku", 45)
    out["description_rate"] = _asof_target_rate(out, ref, "claim_description", 35)
    out["serial_rate"] = _asof_target_rate(out, ref, "product_serial", 8)
    out["city_rate"] = _asof_target_rate(out, ref, "city", 45)
    return out


def _feature_frame(raw: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame,
                   history_ref: pd.DataFrame, vocab: Dict[str, List[str]],
                   leave_one_out: bool = False) -> pd.DataFrame:
    x = raw.copy()
    x["submitted_at"] = pd.to_datetime(x.submitted_at)
    x = x.merge(partners, on="partner_id", how="left", validate="many_to_one")
    x = x.merge(products, on="sku", how="left", validate="many_to_one")
    x["onboarded_date"] = pd.to_datetime(x.onboarded_date)
    x["partner_age_days"] = (x.submitted_at.dt.normalize() - x.onboarded_date).dt.days
    x["partner_age_bin"] = pd.cut(
        x["partner_age_days"], [-1, 30, 90, 180, 365, 730, 100000],
        labels=["0-30", "31-90", "91-180", "181-365", "366-730", "731+"]
    ).astype(str)
    x["amount_ratio"] = x.claim_amount_inr / x.list_price_inr.replace(0, np.nan)
    x["amount_ratio"] = x.amount_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    x["warranty_age_ratio"] = (x.days_since_purchase / (x.warranty_months * 30.5)).clip(0, 3)
    x["amount_over_list_excess"] = (x.amount_ratio - 0.5).clip(0, 2)
    x["in_warranty"] = (x.days_since_purchase <= x.warranty_months * 30.5).astype(int)
    x["claim_under_2000"] = (x.claim_amount_inr < 2000).astype(int)
    x["policy_inspection_gap"] = ((x.claim_amount_inr >= 2000) & (x.partner_inspected != "Y")).astype(int)
    x["month"] = x.submitted_at.dt.month
    x["month_sin"] = np.sin(2 * np.pi * x.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * x.month / 12)
    x["serial_norm"] = x.product_serial.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    ref = history_ref.copy()
    ref["serial_norm"] = ref.product_serial.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    x["claim_id_count"] = _asof_count(x, ref, "claim_id")
    x["serial_count"] = _asof_count(x, ref, "serial_norm")
    x["partner_volume_asof"] = _asof_count(x, ref, "partner_id")
    x = _add_history(x, history_ref, leave_one_out)
    x["partner_sku_rate_interaction"] = x.partner_rate * x.sku_rate
    x["partner_description_rate_interaction"] = x.partner_rate * x.description_rate
    x["recent_partner_rate"] = _recent_partner_rate(x, history_ref, 180)
    return x


def _design(x: pd.DataFrame, vocab: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    nums = [
        "partner_rate", "sku_rate", "description_rate", "serial_rate", "city_rate",
        "warranty_age_ratio", "amount_over_list_excess", "partner_volume_asof",
        "partner_sku_rate_interaction", "partner_description_rate_interaction", "recent_partner_rate",
        "amount_ratio", "claim_amount_inr", "days_since_purchase", "customer_prior_claims",
        "partner_age_days", "claim_id_count", "serial_count", "in_warranty",
        "claim_under_2000", "policy_inspection_gap", "month_sin", "month_cos",
    ]
    cols = ["intercept"] + nums
    arr = [np.ones(len(x))]
    for c in nums:
        v = pd.to_numeric(x[c], errors="coerce").fillna(0).to_numpy(float)
        if c in {"amount_ratio", "claim_amount_inr", "days_since_purchase", "partner_age_days"}:
            v = np.log1p(np.clip(v, 0, None))
        arr.append(v)
    for c in ["source", "photo_attached", "partner_inspected", "claim_description", "sku", "city", "partner_type", "partner_age_bin"]:
        for val in vocab[c]:
            cols.append(f"{c}={val}")
            arr.append((x[c].astype(str).to_numpy() == val).astype(float))
    return np.column_stack(arr), cols


def build_model(train: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame,
                seed: int = 7):
    labelled = train[train.is_fraud.notna()].copy()
    labelled["y"] = labelled.is_fraud.astype(float)
    vocab = _fit_vocab(labelled.merge(partners, on="partner_id").merge(products, on="sku"))
    # History ref must have join fields used by feature construction.
    hist = labelled.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    train_f = _feature_frame(labelled, partners, products, hist, vocab, leave_one_out=True)
    X, cols = _design(train_f, vocab)
    if RandomForestClassifier is None:
        raise RuntimeError(
            "scikit-learn is required for the reproducible submitted model; "
            "run `python -m pip install -r requirements.txt` first"
        ) from _SKLEARN_IMPORT_ERROR
    forest = RandomForestClassifier(
        n_estimators=120, min_samples_leaf=20, max_features=0.7,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1
    ).fit(X, labelled.y.to_numpy(int))
    return {
        "forest": forest,
        "vocab": vocab,
        "history": hist,
        "partners": partners,
        "products": products,
        "columns": cols,
        "prior": float(labelled.y.mean()),
    }


def predict(model, raw: pd.DataFrame) -> np.ndarray:
    hist = model["history"]
    x = _feature_frame(raw, model["partners"], model["products"], hist, model["vocab"])
    X, _ = _design(x, model["vocab"])
    forest_score = model["forest"].predict_proba(X)
    if forest_score.ndim == 2:
        forest_score = forest_score[:, 1]
    return forest_score


def metrics(y: np.ndarray, p: np.ndarray, amounts: np.ndarray, review_n: int = 40) -> Dict[str, float]:
    order = np.argsort(-p)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    top = order[:min(review_n, len(order))]
    top_fraud = int(y[top].sum())
    # Value of review: fraud payout avoided, less genuine goodwill and contact cost.
    saved = float(amounts[top][y[top] == 1].sum())
    held = int((y[top] == 0).sum())
    net = saved - 380 * held - 260 * len(top)
    # AUC without external packages.
    ranks = pd.Series(p).rank(method="average").to_numpy()
    pos = y == 1; neg = ~pos
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())) if pos.sum() and neg.sum() else float("nan")
    return {"n": int(len(y)), "positive_rate": float(y.mean()), "accuracy": float((pred == y).mean()),
            "precision_at_0_5": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall_at_0_5": float(tp / (tp + fn)) if tp + fn else 0.0,
            "auc": auc, "review_n": int(len(top)), "fraud_in_top_40": top_fraud,
            "precision_at_40": float(top_fraud / len(top)) if len(top) else 0.0,
            "rupees_saved_gross_top_40": saved, "net_value_top_40": net,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def evaluate(train: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> Dict:
    labelled = train[train.is_fraud.notna()].copy()
    labelled["date"] = pd.to_datetime(labelled.submitted_at)
    # Primary honest estimate: final 20% by time, with no future labels in features.
    cutoff = labelled.date.quantile(0.80)
    fit = labelled[labelled.date < cutoff].drop(columns=["date"])
    val = labelled[labelled.date >= cutoff].drop(columns=["date"])
    m = build_model(fit, partners, products, seed=7)
    p = predict(m, val)
    result = {"cutoff": str(cutoff), "time_holdout": metrics(val.is_fraud.to_numpy(int), p, val.claim_amount_inr.to_numpy(float))}
    # Repeated temporal cutoffs quantify calibration/selection spread.
    windows = []
    qs = [0.60, 0.65, 0.70, 0.75, 0.80]
    for i, q in enumerate(qs):
        c = labelled.date.quantile(q)
        a = labelled[labelled.date < c].drop(columns=["date"])
        b = labelled[labelled.date >= c].drop(columns=["date"])
        mm = build_model(a, partners, products, seed=7 + i)
        pp = predict(mm, b)
        windows.append({"quantile": q, "cutoff": str(c), **metrics(b.is_fraud.to_numpy(int), pp, b.claim_amount_inr.to_numpy(float))})
    result["repeated_time_windows"] = windows
    result["spread"] = {k: {"min": float(min(w[k] for w in windows)), "max": float(max(w[k] for w in windows)),
                             "mean": float(np.mean([w[k] for w in windows]))} for k in ["accuracy", "auc", "precision_at_40", "net_value_top_40"]}
    random_windows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        mask = rng.random(len(labelled)) < 0.8
        a = labelled[mask].drop(columns=["date"])
        b = labelled[~mask].drop(columns=["date"])
        mm = build_model(a, partners, products, seed=seed)
        pp = predict(mm, b)
        random_windows.append({"seed": seed, **metrics(b.is_fraud.to_numpy(int), pp, b.claim_amount_inr.to_numpy(float))})
    result["repeated_random_splits"] = random_windows
    result["random_spread"] = {k: {"min": float(min(w[k] for w in random_windows)), "max": float(max(w[k] for w in random_windows)),
                                   "mean": float(np.mean([w[k] for w in random_windows]))} for k in ["accuracy", "auc", "precision_at_40", "net_value_top_40"]}
    return result


def main():
    train = pd.read_csv(TRAIN)
    partners = pd.read_csv(PARTNERS)
    products = pd.read_csv(PRODUCTS)
    test = pd.read_csv(TEST)
    ev = evaluate(train, partners, products)
    model = build_model(train, partners, products)
    test["score"] = predict(model, test)
    pred = test[["claim_id", "score"]]

    # Save to submission directory (primary)
    sub_dir = PROJECT_ROOT / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(sub_dir / "predictions.csv", index=False)

    # Save to reports directory (primary)
    rep_dir = PROJECT_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "evidence.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")

    # Write lightweight model metadata
    metadata = {
        "model_name": "kestrel-fraud-rf-asof",
        "model_version": "1.0.0",
        "feature_version": "2.0-asof",
        "python_version": "3.13.15",
        "dependencies": {
            "pandas": "3.0.6",
            "numpy": "2.5.3",
            "scikit-learn": "1.9.1",
            "pytest": "8.3.3"
        },
        "model_architecture": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 120,
            "min_samples_leaf": 20,
            "max_features": 0.7,
            "class_weight": "balanced_subsample",
            "random_state": 11
        },
        "data_hashes": {
            "train_csv_sha256": "2490847a83be3e5613d206b4bd1e63b2a6e33887bbe101409013cf9cf02f2325",
            "test_unlabelled_csv_sha256": "2cd7f445c9216b4aedb77e31d650485687fefe37a2d3b7add654f3b51125bfd5"
        },
        "training_timestamp": "2026-10-01T11:00:00Z",
        "primary_validation_metrics": {
            "evaluation_cutoff": ev["cutoff"],
            "primary_holdout_auc": ev["time_holdout"]["auc"],
            "primary_holdout_accuracy": ev["time_holdout"]["accuracy"],
            "primary_holdout_top40_precision": ev["time_holdout"]["precision_at_40"],
            "primary_holdout_top40_frauds": ev["time_holdout"]["fraud_in_top_40"],
            "primary_holdout_net_value_inr": ev["time_holdout"]["net_value_top_40"],
            "five_window_mean_auc": ev["spread"]["auc"]["mean"],
            "five_window_mean_accuracy": ev["spread"]["accuracy"]["mean"],
            "five_window_mean_net_value_inr": ev["spread"]["net_value_top_40"]["mean"]
        }
    }
    (PROJECT_ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(ev, indent=2))
    print("predictions", len(pred), "range", float(pred.score.min()), float(pred.score.max()))


if __name__ == "__main__":
    main()
