# Kestrel Home — Warranty Claim Fraud Prioritization Service

A production-ready, leakage-safe machine learning system and local review service built for Kestrel Home's warranty operations desk. The system prioritizes incoming warranty claims for an investigation capacity of 40 claims per month, maximizing avoided fraudulent payouts while safeguarding genuine customer goodwill.

---

## Architecture & Project Structure

```text
kestrel-fraud/
│
├── src/
│   ├── model.py                     # Deterministic as-of model pipeline and evaluation suite
│   ├── service.py                   # Local HTTP review API (POST /api/score, GET /api/health)
│   └── app.html                     # Responsive single-page investigator review interface
│
├── tests/
│   ├── test_features.py             # Temporal leakage & causality tests
│   ├── test_model.py                # Schema, validity, batch parity & missing value tests
│   └── test_service.py              # Review API, signals & validation tests
│
├── data/
│   ├── raw/
│   │   ├── train.csv                # Training claims dataset
│   │   └── test_unlabelled.csv      # Unlabelled test claims dataset
│   └── reference/
│       ├── partners.csv             # Partner onboarding reference
│       ├── products.csv             # Product catalog & pricing reference
│       ├── ops-policy.pdf           # Operational policy specification
│       ├── email-thread.txt         # Stakeholder communications & constraints
│       └── sample_submission.csv    # Target schema specification
│
├── reports/
│   ├── evidence.json                # Authoritative validation and benchmark metrics
│   ├── evidence.md                  # Comprehensive empirical validation report
│   ├── feature_experiment.json      # Isolated feature ablation results
│   └── sklearn_benchmark.json       # Model family bake-off results
│
├── submission/
│   ├── predictions.csv              # Authoritative predictions for test claims (2,252 rows)
│   ├── memo-to-ritu.md              # Executive briefing for Head of D2C Operations
│   ├── submission-form.md           # Completed technical and operational submission form
│   └── recording-script.md          # 3-minute presentation and walkthrough script
│
├── model_metadata.json              # Lightweight production model & run metadata
├── requirements.txt                 # Pinned dependencies (pandas, numpy, scikit-learn, pytest)
├── .gitignore                       # Git exclusions (Python bytecode, caches, envs)
└── README.md                        # Authoritative documentation and operational runbook
```

---

## Authoritative Validation Metrics

All metrics reflect a single authoritative, reproducible execution of `python src/model.py`:

| Evaluation Metric | Primary Holdout ($N=2,363$) | 5 Chronological Windows (Mean) | 5 Chronological Windows (Range) |
|---|---|---|---|
| **ROC-AUC** | **0.523** | **0.592** | 0.496 – 0.732 |
| **Accuracy (at 0.5)** | **97.84%** | **98.25%** | 97.84% – 98.48% |
| **Frauds in Top 40** | **5 / 40** | **12.0 / 40** | 5 – 19 |
| **Precision@40** | **12.5%** | **30.0%** | 12.5% – 47.5% |
| **Recall@40** | **10.42%** | **19.85%** | 10.42% – 27.54% |
| **Gross Payout Avoided** | **Rs 45,338** | **Rs 92,078** | Rs 45,338 – Rs 142,330 |
| **Net Economic Value** | **Rs 21,638** | **Rs 71,038** | Rs 21,638 – Rs 123,950 |

*Net value incorporates Rs 380 goodwill penalty per genuine claim held and Rs 260 contact cost per review ($40 \times \text{Rs 260} = \text{Rs 10,400}$).*

---

## Business Conclusion: Model vs. Heuristics

On the primary latest holdout window ($N=2,363$, 48 true frauds):
- **Random Selection:** Captures 0.81 frauds on average (2.03% precision; net value **Rs -23,037**).
- **Model Top-40 Queue:** Captures **5 frauds** (12.5% precision; net value **Rs 21,638**).
- **Claim-Amount Heuristic:** Captures **4 frauds** (10.0% precision; net value **Rs 35,161**).

### Strategic Takeaways
1. **Incremental Fraud Detection:** The model demonstrates incremental fraud capture over the simple amount heuristic (+1 additional fraud case, a 25% detection gain), but has not yet demonstrated superior economic value on this single window because the amount heuristic's 4 frauds had larger individual claim sizes.
2. **Why Retain for Shadow Mode:** Across the full multi-window timeline, the model averaged **Rs 71,038** net value vs. **Rs 47,899** for the amount heuristic. Furthermore, a simple amount heuristic is easily gamed by fraudsters splitting claims, whereas the model captures multi-signal structural risk (partner rate history, serial reuse, policy inspection non-compliance, amount-to-list-price ratios).
3. **Objective Optimization:** To maximize rupees recovered, future queue ranking should sort by **Expected Loss Avoided**:
   $$\text{Priority Score} = P(\text{fraud}) \times \text{claim\_amount\_inr} - (1 - P(\text{fraud})) \times \text{goodwill\_cost}$$

---

## Quickstart & Replication

### 1. Environment Setup
Requires Python 3.10+ (tested on Python 3.13.15) and standard pinned dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```
All 16 tests verify leakage-safety, batch parity, missing value robustness, and API integrity.

### 3. Generate Authoritative Predictions, Evidence & Metadata
```bash
python src/model.py
```
This regenerates `submission/predictions.csv`, `reports/evidence.json`, and `model_metadata.json` deterministically.

### 4. Launch Local Review Service
```bash
python src/service.py
```
Then navigate to `http://127.0.0.1:8000/` in your browser.

---

## API Specifications

### `GET /api/health`
Health check endpoint.
```json
{
  "ok": true,
  "model": "leakage-safe scikit-learn random forest ranker"
}
```

### `POST /api/score`
Scores an incoming warranty claim and returns human-interpretable review signals.

**Request:**
```json
{
  "claim_id": "WC711348",
  "submitted_at": "2026-07-15T10:00",
  "partner_id": "SP3207",
  "sku": "KH-AF-01",
  "product_serial": "KH123456789",
  "days_since_purchase": 120,
  "claim_amount_inr": 3500,
  "photo_attached": "Y",
  "partner_inspected": "Y",
  "claim_description": "motor not running",
  "customer_prior_claims": 1,
  "source": "crm",
  "inspector_note": ""
}
```

**Response:**
```json
{
  "ok": true,
  "claim_id": "WC711348",
  "fraud_score": 0.064811,
  "review_priority": "rank in the monthly batch",
  "review_capacity": "40 claims per month",
  "reasons": [
    "Review signal: partner has been onboarded for 530 days; historical partner behaviour is included.",
    "Review signal: claim amount is 69% of list price, a moderate relative amount.",
    "Review signal: customer has 1 prior claim(s)."
  ],
  "note": "This is a prioritization score, not a calibrated probability. Final review priority requires ranking the monthly batch; reasons are relevant review/policy signals, not causal feature explanations."
}
```

---

## Data Hygiene & Security Safeguards

- **Strictly As-Of Features:** All historical partner rates, serial counts, and trend features are calculated strictly using reference rows timestamped prior to each claim, preventing future-period data leakage.
- **Undecided Outcomes:** 215 unresolved claims with blank investigation outcomes are excluded from training rather than falsely assumed genuine.
- **Untrusted Free Text:** Injected prompt instructions within `claim_description` are treated strictly as untrusted data values and never executed.
- **Graceful Fallbacks:** Claims referencing unseen partner IDs or SKUs fall back to smoothed population priors and return explicit explanatory signals. Missing required fields produce structured 400 Bad Request responses rather than server crashes.

