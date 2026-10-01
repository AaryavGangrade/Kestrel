"""Tests for the local review service API and single-claim scoring logic."""
from __future__ import annotations

import json
import sys
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from service import Handler, score_record


@pytest.fixture(scope="module")
def valid_record():
    return {
        "claim_id": "TEST-CLAIM-001",
        "submitted_at": "2026-06-15 14:30:00",
        "partner_id": "SP3207",
        "sku": "KB-MEC-RGB",
        "product_serial": "KB-MEC-RGB-9921",
        "days_since_purchase": 45,
        "claim_amount_inr": 1200,
        "photo_attached": "Y",
        "partner_inspected": "Y",
        "claim_description": "Key switch failure on spacebar",
        "customer_prior_claims": 0,
        "source": "partner_portal",
    }


def test_score_record_success(valid_record):
    """Verify normal valid claim produces score, review metadata, and signals."""
    res = score_record(valid_record)
    assert res["ok"] is True
    assert "fraud_score" in res
    assert 0.0 <= res["fraud_score"] <= 1.0
    assert res["review_capacity"] == "40 claims per month"
    assert isinstance(res["reasons"], list)
    assert len(res["reasons"]) > 0


def test_score_record_missing_required_field(valid_record):
    """Verify missing required fields return ok=False with explicit field list."""
    incomplete = valid_record.copy()
    del incomplete["claim_id"]
    del incomplete["claim_amount_inr"]

    res = score_record(incomplete)
    assert res["ok"] is False
    assert "Missing required fields" in res["error"]
    assert "claim_id" in res["missing_fields"]
    assert "claim_amount_inr" in res["missing_fields"]


def test_score_record_unknown_partner_fallback(valid_record):
    """Verify unknown partner uses fallback and informs investigator in reasons."""
    unknown = valid_record.copy()
    unknown["partner_id"] = "UNSEEN_PARTNER_9999"

    res = score_record(unknown)
    assert res["ok"] is True
    assert 0.0 <= res["fraud_score"] <= 1.0
    reasons_text = " ".join(res["reasons"])
    assert "Partner ID is not in the reference file" in reasons_text


def test_score_record_unknown_sku_fallback(valid_record):
    """Verify unknown SKU uses fallback and informs investigator in reasons."""
    unknown = valid_record.copy()
    unknown["sku"] = "UNSEEN_SKU_NONEXISTENT"

    res = score_record(unknown)
    assert res["ok"] is True
    reasons_text = " ".join(res["reasons"])
    assert "SKU is not in the product reference" in reasons_text


def test_score_record_missing_serial_signal(valid_record):
    """Verify missing serial produces review signal without failing."""
    no_serial = valid_record.copy()
    no_serial["product_serial"] = ""

    res = score_record(no_serial)
    assert res["ok"] is True
    reasons_text = " ".join(res["reasons"])
    assert "product serial is missing" in reasons_text


@pytest.fixture(scope="module")
def live_server():
    """Start local test HTTP server on an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_api_health(live_server):
    """Verify GET /api/health endpoint returns 200 and model metadata."""
    req = Request(f"{live_server}/api/health", method="GET")
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        assert "scikit-learn" in data["model"]


def test_api_score_post(live_server, valid_record):
    """Verify POST /api/score returns JSON response with fraud score."""
    req = Request(
        f"{live_server}/api/score",
        data=json.dumps(valid_record).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        assert "fraud_score" in data


def test_api_html_ui(live_server):
    """Verify GET / serves investigator HTML UI."""
    req = Request(f"{live_server}/", method="GET")
    with urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<!DOCTYPE html>" in content or "<html" in content
        assert "Kestrel" in content
