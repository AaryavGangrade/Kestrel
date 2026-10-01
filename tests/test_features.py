"""Tests for feature engineering correctness and temporal leakage prevention."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import _asof_count, _asof_target_rate, _recent_partner_rate


def test_asof_target_rate_no_future_leakage():
    """Verify that a claim never uses target information from the same or future timestamps."""
    ref = pd.DataFrame({
        "partner_id": ["P1", "P1", "P1"],
        "submitted_at": [
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
        ],
        "y": [1, 1, 1],
    })

    rows = pd.DataFrame({
        "partner_id": ["P1", "P1"],
        "submitted_at": [
            "2026-01-01 10:00:00",  # Same timestamp as first ref row
            "2026-01-02 09:00:00",  # Between row 1 and row 2
        ],
    })

    # At 2026-01-01 10:00:00, allow_exact_matches=False ensures 0 prior claims are visible
    rates = _asof_target_rate(rows, ref, "partner_id", strength=10.0)
    # Default smoothed prior is 0.01: (0 + 10 * 0.01) / (0 + 10) = 0.01
    assert np.isclose(rates[0], 0.01), f"Expected prior 0.01, got {rates[0]}"

    # At 2026-01-02 09:00:00, only the 2026-01-01 10:00:00 event is visible (y=1)
    # (1 + 10 * 0.01) / (1 + 10) = 1.1 / 11 = 0.10
    assert np.isclose(rates[1], 1.1 / 11.0), f"Expected 0.10, got {rates[1]}"


def test_asof_target_rate_label_invariance_to_future_changes():
    """Altering future labels must have zero impact on earlier rows' features."""
    ref1 = pd.DataFrame({
        "partner_id": ["P1", "P1", "P1"],
        "submitted_at": [
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
        ],
        "y": [0, 0, 0],
    })

    ref2 = pd.DataFrame({
        "partner_id": ["P1", "P1", "P1"],
        "submitted_at": [
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
        ],
        "y": [0, 1, 1],  # Future outcomes flipped to 1
    })

    query = pd.DataFrame({
        "partner_id": ["P1"],
        "submitted_at": ["2026-01-02 09:00:00"],  # Before row 2 and row 3
    })

    rate1 = _asof_target_rate(query, ref1, "partner_id", strength=25.0)[0]
    rate2 = _asof_target_rate(query, ref2, "partner_id", strength=25.0)[0]

    assert np.isclose(rate1, rate2), "Future label changes leaked into historical feature!"


def test_asof_count_strict_precedence():
    """Verify that as-of counts strictly ignore concurrent or future rows."""
    ref = pd.DataFrame({
        "partner_id": ["P1", "P1", "P1", "P2"],
        "submitted_at": [
            "2026-01-01 12:00:00",
            "2026-01-01 12:00:00",  # Concurrent row
            "2026-01-05 12:00:00",
            "2026-01-01 08:00:00",
        ],
    })

    query = pd.DataFrame({
        "partner_id": ["P1", "P1", "P1"],
        "submitted_at": [
            "2026-01-01 12:00:00",  # Exact match with first two
            "2026-01-02 12:00:00",  # After first two, before third
            "2026-01-06 12:00:00",  # After all three
        ],
    })

    counts = _asof_count(query, ref, "partner_id")
    assert counts[0] == 0, f"Concurrent event leaked: expected count 0, got {counts[0]}"
    assert counts[1] == 2, f"Expected count 2, got {counts[1]}"
    assert counts[2] == 3, f"Expected count 3, got {counts[2]}"


def test_recent_partner_rate_causality():
    """Verify exponential weighting ignores future and contemporaneous observations."""
    hist = pd.DataFrame({
        "partner_id": ["P1", "P1"],
        "submitted_at": ["2026-01-01 00:00:00", "2026-01-10 00:00:00"],
        "y": [1, 1],
    })

    rows = pd.DataFrame({
        "partner_id": ["P1", "P1"],
        "submitted_at": [
            "2026-01-01 00:00:00",  # Same as hist[0]
            "2026-01-05 00:00:00",  # Between hist[0] and hist[1]
        ],
    })

    rates = _recent_partner_rate(rows, hist, days=180)
    # First row has age <= 0 for all hist rows -> weight 0 -> prior 0.01
    assert np.isclose(rates[0], 0.01)
    # Second row only sees hist[0] (age = 4 days > 0)
    assert rates[1] > 0.01
