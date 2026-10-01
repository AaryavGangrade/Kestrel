"""Tests for model building, prediction output schema, parity, and robustness."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import PARTNERS, PRODUCTS, TEST, TRAIN, build_model, predict


@pytest.fixture(scope="module")
def model_and_data():
    train = pd.read_csv(TRAIN)
    partners = pd.read_csv(PARTNERS)
    products = pd.read_csv(PRODUCTS)
    test = pd.read_csv(TEST)
    model = build_model(train, partners, products)
    return model, test, partners, products


def test_predictions_schema_and_integrity():
    """Verify predictions.csv exists, has exact columns, and matches test claim IDs."""
    pred_path = PROJECT_ROOT / "submission" / "predictions.csv"
    assert pred_path.exists(), f"Missing predictions file at {pred_path}"

    df = pd.read_csv(pred_path)
    # Exact required columns
    assert list(df.columns) == ["claim_id", "score"], f"Unexpected columns: {df.columns}"

    # Exactly 2,252 rows
    assert len(df) == 2252, f"Expected 2252 rows, found {len(df)}"

    # Match test claim IDs exactly
    test_df = pd.read_csv(TEST)
    assert list(df["claim_id"]) == list(test_df["claim_id"]), "Claim IDs do not match test unlabelled data"
    assert df["claim_id"].nunique() == 2252, "Duplicate claim IDs detected"
    assert not df["score"].isna().any(), "NaN values found in predictions"


def test_score_validity_range():
    """Verify that predictions are valid floats between 0.0 and 1.0 with positive variance."""
    pred_path = PROJECT_ROOT / "submission" / "predictions.csv"
    df = pd.read_csv(pred_path)

    scores = df["score"].to_numpy(dtype=float)
    assert np.all(scores >= 0.0), f"Scores below 0.0 found: min={scores.min()}"
    assert np.all(scores <= 1.0), f"Scores above 1.0 found: max={scores.max()}"
    assert np.var(scores) > 1e-6, "Scores are degenerate/constant"


def test_batch_vs_single_record_parity(model_and_data):
    """Verify scoring a claim individually produces the same score as scoring in a batch."""
    model, test, _, _ = model_and_data

    # Sample representative rows across the test set
    sample = test.iloc[:20].copy()
    batch_scores = predict(model, sample)

    single_scores = []
    for i in range(len(sample)):
        single_row = sample.iloc[[i]].copy()
        score = predict(model, single_row)[0]
        single_scores.append(score)

    single_scores = np.array(single_scores)
    max_diff = np.max(np.abs(batch_scores - single_scores))
    assert max_diff < 1e-5, f"Discrepancy between batch and single-row scoring: max diff={max_diff}"


def test_unknown_and_missing_values_robustness(model_and_data):
    """Verify model gracefully handles unseen partners, SKUs, and missing serials."""
    model, test, _, _ = model_and_data

    sample = test.iloc[:5].copy()
    sample.loc[0, "partner_id"] = "PARTNER_UNSEEN_99999"
    sample.loc[1, "sku"] = "SKU_UNSEEN_NONEXISTENT"
    sample.loc[2, "product_serial"] = ""
    sample.loc[3, "product_serial"] = np.nan
    sample.loc[4, "claim_description"] = "UNSEEN_ANOMALOUS_DESCRIPTION_999"

    scores = predict(model, sample)
    assert len(scores) == 5
    assert not np.isnan(scores).any(), "NaN score produced on unknown/missing inputs"
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
