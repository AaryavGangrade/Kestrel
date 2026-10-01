"""Local HTTP service for one-claim review.

Run with the bundled/runtime Python or any environment with pandas and numpy:
    python service.py
Then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import PARTNERS, PRODUCTS, TRAIN, build_model, predict

ROOT = SRC_DIR
TRAIN_DF = pd.read_csv(TRAIN)
PARTNER_DF = pd.read_csv(PARTNERS)
PRODUCT_DF = pd.read_csv(PRODUCTS)
MODEL = build_model(TRAIN_DF, PARTNER_DF, PRODUCT_DF)


def _record_reasons(record: dict) -> list[str]:
    reasons = []
    partner = PARTNER_DF[PARTNER_DF.partner_id.astype(str) == str(record.get("partner_id", ""))]
    product = PRODUCT_DF[PRODUCT_DF.sku.astype(str) == str(record.get("sku", ""))]
    if partner.empty:
        reasons.append("Partner ID is not in the reference file; using the population fallback.")
    else:
        onboarded = pd.to_datetime(partner.iloc[0].onboarded_date)
        submitted = pd.to_datetime(record.get("submitted_at"))
        age = (submitted.normalize() - onboarded).days
        if age < 180:
            reasons.append(f"Review signal: partner was onboarded {max(age, 0)} days before this claim.")
        else:
            reasons.append(f"Review signal: partner has been onboarded for {age} days; historical partner behaviour is included.")
    if product.empty:
        reasons.append("SKU is not in the product reference; using the population fallback.")
    else:
        price = float(product.iloc[0].list_price_inr)
        amount = float(record.get("claim_amount_inr", 0) or 0)
        ratio = amount / price if price else 0
        if ratio >= 0.75:
            reasons.append(f"Review signal: claim amount is {ratio:.0%} of list price, a high relative amount.")
        elif ratio >= 0.5:
            reasons.append(f"Review signal: claim amount is {ratio:.0%} of list price, a moderate relative amount.")
    if int(record.get("customer_prior_claims", 0) or 0) > 0:
        reasons.append(f"Review signal: customer has {int(record['customer_prior_claims'])} prior claim(s).")
    if str(record.get("product_serial", "")).strip() == "":
        reasons.append("Data-quality signal: product serial is missing; serial-based checks are unavailable.")
    if str(record.get("claim_id", "")).strip() == "":
        reasons.append("Data-quality signal: claim ID is missing; duplicate-claim checks are unavailable.")
    amount = float(record.get("claim_amount_inr", 0) or 0)
    inspected = str(record.get("partner_inspected", ""))
    submitted = pd.to_datetime(record.get("submitted_at"))
    if submitted >= pd.Timestamp("2026-05-01") and amount >= 2000 and inspected != "Y":
        reasons.append("Policy gap: claim is at least Rs 2,000 after 1 May 2026 without partner inspection sign-off.")
    if inspected != "Y":
        reasons.append("Policy/data signal: partner inspection sign-off is absent.")
    if not reasons:
        reasons.append("No single hard rule fired; use the score to prioritize the limited review queue.")
    return reasons[:5]


def score_record(record: dict) -> dict:
    row = pd.DataFrame([record])
    # Complete missing optional fields without masking missing-key behavior.
    for c in ["inspector_note"]:
        if c not in row:
            row[c] = ""
    missing = [c for c in ["claim_id", "submitted_at", "partner_id", "sku", "product_serial",
                           "days_since_purchase", "claim_amount_inr", "photo_attached",
                           "partner_inspected", "claim_description", "customer_prior_claims", "source"]
               if c not in row or pd.isna(row.iloc[0].get(c))]
    if missing:
        return {"ok": False, "error": "Missing required fields", "missing_fields": missing}
    score = float(predict(MODEL, row)[0])
    return {"ok": True, "claim_id": record.get("claim_id"), "fraud_score": round(score, 6),
            "review_priority": "rank in the monthly batch",
            "review_capacity": "40 claims per month",
            "reasons": _record_reasons(record),
            "note": "This is a prioritization score, not a calibrated probability. Final review priority requires ranking the monthly batch; reasons are relevant review/policy signals, not causal feature explanations."}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, content_type="application/json"):
        body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/api/health":
            self._send(200, {"ok": True, "model": "leakage-safe scikit-learn random forest ranker"}); return
        if urlparse(self.path).path in ("/", "/index.html"):
            self._send(200, (ROOT / "app.html").read_text(encoding="utf-8"), "text/html; charset=utf-8"); return
        self._send(404, {"ok": False, "error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/score":
            self._send(404, {"ok": False, "error": "Not found"}); return
        try:
            length = int(self.headers.get("Content-Length", "0")); record = json.loads(self.rfile.read(length))
            out = score_record(record); self._send(200 if out.get("ok") else 400, out)
        except Exception as exc:
            self._send(400, {"ok": False, "error": "Could not score record", "detail": str(exc)})


if __name__ == "__main__":
    print("Kestrel review service: http://127.0.0.1:8000")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
