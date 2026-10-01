# Evidence and Empirical Validation Report

## A. Business Problem

Kestrel Home faces potential warranty fraud across an expanding network of service partners and direct-to-consumer product lines. However, fraudulent claims represent only approximately 1.23% of labelled historical claims. Furthermore, the operations investigation desk operates under a strict capacity constraint of reviewing roughly 40 claims per month.

Crucially, an investigation hold carries real business costs: ₹380 in customer goodwill penalty for each genuine claim delayed, and ₹260 in customer contact and administrative overhead per claim reviewed. Therefore, the system is designed strictly as a **fraud-risk scoring and review-ranking aid** to prioritize this 40-claim human review queue—not as an autonomous rejection system or a calibrated probability generator.

## B. Data and Policy Constraints

- **Labelled Dataset:** 12,029 total training records; 11,814 labelled claims (145 fraud cases, 1.23% base rate) and 215 blank/unresolved outcomes excluded from supervised training rather than assumed genuine.
- **Test Dataset:** 2,252 unlabelled test claims; `predictions.csv` produces exactly 2,252 rows matching the required schema.
- **Identifier Handling:** 681 repeated claim-ID groups exist in training (representing resubmissions, not independent claims); claim IDs are not memorized as labels. Serials repeat across claims and are normalized and tracked via cumulative counts.
- **Unseen Entities:** 14 partners appearing in the test period have no prior labelled claims. The model and review service apply a conservative smoothed population prior (1% baseline) for unseen partners and SKUs.
- **Policy Thresholds:** Operational policy mandates partner inspection sign-off for claims ≥ ₹2,000 submitted on or after 1 May 2026. Non-compliant claims are flagged with an explicit policy signal.
- **Untrusted Free Text:** Injected prompt instructions within claim descriptions are treated strictly as untrusted string literals and never executed.

## C. Leakage Prevention and As-Of Feature Construction

To ensure deployment integrity, all historical statistics are computed strictly as-of each claim's submission timestamp:
- `_asof_target_rate()`: Historical smoothed fraud rates for partners, SKUs, descriptions, serials, and cities use backward-looking point-in-time joins (`pd.merge_asof(..., direction="backward", allow_exact_matches=False)`), preventing concurrent or future labels from leaking into historical features.
- `_asof_count()`: Cumulative claim and serial occurrence counts strictly exclude concurrent or future observations.
- `_recent_partner_rate()`: Exponentially weighted 180-day partner fraud rate applies zero weight to events with non-positive time deltas ($age \le 0$). This feature is retained as part of the final validated feature pipeline.

## D. Chronological Validation

Because fraud behavior evolves over time, random train/test splits produce optimistic results (random 80/20 splits yielded AUC 0.790–0.941, mean 0.879). Chronological validation simulates actual deployment by training on history before a cutoff date and testing on subsequent claims.

The primary chronological holdout evaluated 2,363 later-labelled claims (48 frauds) submitted after 28 March 2026:

| Evaluation Check | Result |
|---|---:|
| Total Evaluation Claims | 2,363 |
| True Fraud Cases | 48 (2.03% prevalence) |
| Accuracy (at default 0.5 threshold) | 97.84% (error rate: 2.16%) |
| Confusion Matrix (at 0.5 threshold) | TP: 5, FP: 8, TN: 2,307, FN: 43 |
| ROC-AUC for continuous ranking | 0.523 |
| Frauds in Top-40 Review Queue | 5 / 40 |
| Precision@40 | 12.5% |
| Recall@40 | 10.42% |
| Gross Fraudulent Payout Avoided | ₹45,338 |
| Genuine Claims Held in Top 40 | 35 |
| Goodwill Penalty (35 × ₹380) | ₹13,300 |
| Investigation Overhead (40 × ₹260) | ₹10,400 |
| Estimated Net Economic Value | ₹21,638 |

Continuous ranking discrimination on this holdout is modest (ROC-AUC 0.523), reflecting distribution shifts in partner cohorts and claim amounts. High accuracy (97.84%) is expected given the 1.23% base rate and is not by itself sufficient evidence of fraud capture.

## E. Model-Selection Discipline

The final model was not selected solely by maximizing AUC on the observed chronological validation windows. These windows are intended to approximate future-period performance, and repeatedly tuning against the same windows could overfit those validation periods. Model selection therefore considered chronological ranking performance together with top-40 precision, estimated net value, operational stability, and implementation complexity. The model was evaluated across multiple chronological windows rather than repeatedly optimized to produce a higher or narrower AUC range. The remaining variation in chronological AUC is treated as a temporal-performance characteristic to monitor on future claims.

## F. Top-40 Business Metrics Across Chronological Windows

Across five sequential chronological windows ($q=0.60, 0.65, 0.70, 0.75, 0.80$):
- **ROC-AUC:** Ranged from 0.496 to 0.732 (mean: 0.592).
- **Accuracy:** Ranged from 97.84% to 98.48% (mean: 98.25%).
- **Precision@40:** Ranged from 12.5% to 47.5% (mean: 30.0%; average of 12.0 frauds caught per 40-claim queue).
- **Estimated Net Value:** Ranged from ₹21,638 to ₹123,950 (mean: ₹71,038).

These figures represent historical validation results across overlapping operational windows, not guaranteed monthly profit forecasts.

## G. Comparison Against Simple Baselines

On the primary latest chronological holdout ($N=2,363$, 48 frauds):
1. **Random 40-Claim Queue:** Expected capture of 0.81 frauds (2.03% precision, 1.69% recall; expected net loss of approximately -₹23,037).
2. **Claim-Amount Heuristic (Top 40 by Claim Amount):** Captured 4 frauds (10.0% precision, 8.33% recall), avoiding ₹59,241 gross payout. After ₹13,680 goodwill penalty (36 genuine holds) and ₹10,400 review overhead, net economic value was ₹35,161.
3. **Random Forest Model Queue:** Captured 5 frauds (12.5% precision, 10.42% recall), avoiding ₹45,338 gross payout, delivering ₹21,638 net economic value.

**Honest Business Comparison:** On this specific latest window, the model captured incremental fraud over the claim-amount heuristic (5 vs. 4 frauds, a 25% detection gain), but the claim-amount heuristic achieved higher net value (₹35,161 vs. ₹21,638) because its caught fraud claims happened to have larger individual claim amounts (averaging ₹14,810 vs. ₹9,068).

The model is retained for review-ranking because:
- Across all five chronological windows, the model averaged ₹71,038 net value (12.0 frauds caught) vs. ₹47,899 (3.8 frauds caught) for the claim-amount heuristic.
- A pure claim-amount rule is easily gamed by fraudulent partners splitting claims just below review thresholds.
- Future ranking can sort directly by **Expected Loss Avoided** ($\text{Score} = P(\text{fraud}) \times \text{claim\_amount} - (1 - P(\text{fraud})) \times \text{goodwill\_cost}$) to optimize financial recovery.

## H. Limitations

- **Temporal Variation:** Fraud patterns, partner cohorts, and claim amounts change over time. Primary holdout fraud prevalence was 2.03% vs. 1.03% earlier in history; mean claim amounts fell from ₹2,691 to ₹2,321.
- **Uncalibrated Output:** Output scores reflect relative risk ranking for queue sorting, not calibrated posterior probabilities.
- **Sample Scarcity:** Individual monthly evaluation windows contain relatively few true fraud cases (e.g., 48 in the primary holdout), resulting in wider variance in point estimates.

## I. Operational Deployment Recommendation

1. **Shadow-Mode Execution:** Deploy the local review service in shadow mode alongside the existing review process for one month without altering customer payouts.
2. **Decision Support:** Provide human reviewers with the model score and explicit explanatory signals (partner history, list-price ratio, inspection compliance, customer claim history).
3. **Ongoing Monitoring:** Track monthly rolling precision@40, net financial recovery, and partner cohort shifts. If rolling net value falls below the simple baseline, re-tune ranking weights or pause automated queue sorting.

## J. Model Bake-off and Benchmarks

Candidate algorithms were benchmarked across the five chronological windows using identical strictly-as-of features:
- **Random Forest:** Mean AUC 0.592, Mean Precision@40 30.0%, Mean Net Value ₹71,038. Selected for ranking stability, multi-signal feature handling, and auditability.
- **Extra Trees:** Mean AUC 0.552, Mean Precision@40 30.0%, Mean Net Value ₹71,038.
- **HistGradientBoosting:** Mean AUC 0.523, Mean Precision@40 30.0%, Mean Net Value ₹71,038.
- **Gradient Boosting:** Mean AUC 0.463, Mean Precision@40 30.0%, Mean Net Value ₹71,038.
- **Logistic Regression:** Mean AUC 0.568, Mean Precision@40 21.0%, Mean Net Value ₹51,347.

The promoted feature set incorporates strictly-as-of partner rates, SKU rates, description rates, city rates, serial counts, partner volume, 180-day recency-weighted partner trends (`recent_partner_rate`), warranty age ratio, and policy inspection gap indicators.

## K. Automated Test Verification

The test suite in `tests/` (`pytest tests/`) executes 16 automated tests verifying implementation hygiene and pipeline integrity:
1. **Temporal leakage prevention:** Verified that as-of target rates and counts never see concurrent or future events, and future label modifications have 0.0 impact on historical records (`test_features.py`).
2. **Output schema & shape:** Verified `submission/predictions.csv` contains exactly 2,252 rows, matching test claim IDs in exact sequence with no duplicates or missing values (`test_model.py`).
3. **Score validity:** Verified continuous scores fall strictly within [0.0, 1.0] with non-zero variance (`test_model.py`).
4. **Batch vs. single-record parity:** Verified that scoring a record individually yields identical scores to batch scoring with diff < 1e-5 (`test_model.py`).
5. **Missing/unknown value resilience:** Verified graceful fallback handling for unseen partners, unseen SKUs, missing serials, and unobserved categories (`test_model.py`).
6. **Local review service API:** Verified `GET /api/health`, `POST /api/score`, missing-field validation errors, and informative employee review signals (`test_service.py`).
