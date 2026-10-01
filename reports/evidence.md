# Evidence that it works

## Data checks

- 12,029 training rows; 11,814 labelled and 215 blank/undecided outcomes excluded from training.
- 145 fraud outcomes: 1.23% of labelled history.
- 2,252 test rows; predictions have exactly 2,252 rows and match the sample submission columns.
- 681 repeated claim-ID groups in train; no claim-ID overlap between train and test.
- 2,449 repeated-serial groups in train; 797 test rows reuse a serial seen in train.
- 14 test-period partners are unseen in labelled training. The service and model use a population fallback for them.
- All partner and SKU reference keys currently resolve; the missing-key path was tested anyway.
- All labelled claims are within the product warranty window after joining the product table, so this field is not allowed to pretend to be a strong separator.

## Validation results

The primary deployment-like check is chronological: train before 28 March 2026 and evaluate the 2,363 later labelled claims.

| Check | Result |
|---|---:|
| Accuracy | 97.84% |
| ROC-AUC for continuous ranking | 0.523 |
| Fraud in top 40 review queue | 5 / 40 (12.5%) |
| Gross fraudulent payout avoided in top 40 | Rs 45,338 |
| Estimated net value after Rs 380 goodwill for genuine holds and Rs 260 contact cost per reviewed claim | Rs 21,638 |

The low chronological AUC is reported plainly: fraud patterns move over time. Accuracy is not sufficient evidence for a fraud queue. The ranking should first run in shadow mode, be compared with investigator outcomes, and be turned off if it does not beat a simple review baseline.

On this latest window, a random 40-claim queue would capture 0.81 frauds on average (2.03% precision and 1.69% recall; 20,000 simulated queues; expected net value about Rs -23,037). A highest-claim-amount queue captured 4 frauds (10.0% precision, 8.33% recall) and produced Rs 35,161 net value (Rs 59,241 gross saved less Rs 13,680 goodwill on 36 genuine holds and Rs 10,400 contact cost). The model captured 5 of 48 frauds (12.5% precision, 10.42% recall) and produced Rs 21,638 net value (Rs 45,338 gross saved less Rs 13,300 goodwill on 35 genuine holds and Rs 10,400 contact cost).

**Explicit business conclusion:** The model demonstrates incremental fraud capture over the simple amount heuristic on the latest window (5 frauds vs 4 frauds, a 25% increase in fraud detection), but has not yet demonstrated superior economic value on this window (Rs 21,638 vs Rs 35,161 net value). The amount heuristic produced higher net value here because its 4 detected frauds averaged Rs 14,810 in gross savings, whereas the model's 5 detected frauds averaged Rs 9,068.

**Why retain the model for shadow-mode testing?**
1. **Multi-window track record:** Across earlier chronological windows (q=0.60, 0.65, 0.70), the model substantially outperformed the amount heuristic on both fraud capture (19 vs 3, 16 vs 4, 12 vs 4) and net value (Rs 123,950 vs Rs 32,635; Rs 94,421 vs Rs 48,299; Rs 80,747 vs Rs 48,299). Across all five windows, the model averaged Rs 71,038 net value (and 12.0 frauds caught) vs Rs 47,899 net value (and 3.8 frauds caught) for the amount heuristic.
2. **Adversarial gaming:** A pure claim-amount heuristic is easily discovered and exploited by fraudulent partners who submit multiple claims just below high-amount cutoffs. The model captures structural fraud signals (partner submission history, serial reuse, policy inspection gaps, claim-to-list-price ratios) across all claim sizes.
3. **Zero-risk empirical baseline:** Running in shadow mode alongside human review allows Kestrel to measure whether live claims benefit from the model's pattern recognition without risking customer goodwill or altering payouts.

**How to optimize the actual business objective:**
Rather than ranking strictly by estimated fraud probability $P(\text{fraud})$, the model queue can be sorted by **Expected Loss Avoided**:
$$\text{Expected Net Value} = P(\text{fraud}) \times \text{claim\_amount\_inr} - (1 - P(\text{fraud})) \times \text{goodwill\_cost}$$
This directly optimizes Farhan Sheikh's financial KPI (rupees saved per claim checked) by prioritizing claims with both high risk and high financial exposure.

Across five chronological cutoffs, accuracy ranged from 97.84% to 98.48% (mean 98.25%), AUC from 0.496 to 0.732 (mean 0.592; latest window 0.496), top-40 precision from 12.5% to 47.5% (mean 30.0%), and net top-40 value from Rs 21,638 to Rs 123,950 (mean Rs 71,038). Across five repeated random 80/20 splits, AUC ranged from 0.790 to 0.941 (mean 0.879), accuracy from 98.86% to 99.41% (mean 99.14%), top-40 precision from 40.0% to 55.0% (mean 46.5%), and net top-40 value from Rs 37,171 to Rs 127,097 (mean Rs 74,210). The gap is a calibration/drift warning, not a number to hide.

## What gets wrong

The difficult cases are recent partner cohorts and partner histories whose fraud rate changes after the training window. Quantitatively, the latest window's fraud prevalence was 2.03% versus 1.03% earlier; mean claim amount fell from Rs 2,691 to Rs 2,321; unique partners rose from 349 to 365; and median days since purchase moved from 213 to 220. This supports a distribution-shift explanation for why random-split AUC is optimistic and the latest chronological ranking is weaker. The model can also be overconfident for a rare partner with only one or two labelled claims; smoothing reduces this but does not remove the uncertainty. A score is therefore not a payout decision.

## Money estimate

Across five historical validation windows, estimated net value ranged from Rs 21,638 to Rs 123,950 (mean Rs 71,038). These windows overlap and are not independent monthly forecasts, so the range is more meaningful than an average monthly-profit claim.

## Model bake-off

External scikit-learn models were compared on the same five chronological windows after the as-of leakage fix: logistic regression, Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting. Random Forest was selected for the best mean chronological ranking, competitive top-40 results, stable operation, and low operational complexity. The final promoted feature set adds warranty-age ratio, claim-amount excess, as-of partner volume, partner interaction rates, and a strictly as-of 180-day partner trend. The isolated feature experiment showed AUC improving from 0.557 to 0.568 while top-40 precision stayed at 30.0% and top-40 net value stayed at Rs 71,038 across its five windows. The exact authoritative production run in `evidence.json` reports mean chronological AUC 0.592, top-40 precision 30.0%, and historical net-value range Rs 21,638-Rs 123,950 (mean Rs 71,038). The feature addition improved statistical ranking, but did not alter the constrained queue metric in that experiment; it was retained because it improved earlier windows, did not reduce queue performance, and remains strictly leakage-safe.


## Test coverage

The test suite in `tests/` (`pytest tests/`) executes 16 automated tests verifying implementation hygiene and pipeline integrity:
1. **Temporal leakage prevention:** verified that as-of target rates and counts never see concurrent or future events, and future label modifications have 0.0 impact on historical records (`test_features.py`).
2. **Output schema & shape:** verified `submission/predictions.csv` contains exactly 2,252 rows, matching test claim IDs in exact sequence with no duplicates or missing values (`test_model.py`).
3. **Score validity:** verified continuous scores fall strictly within [0.0, 1.0] with non-zero variance (`test_model.py`).
4. **Batch vs. single-record parity:** verified that scoring a record individually yields identical scores to batch scoring with diff < 1e-5 (`test_model.py`).
5. **Missing/unknown value resilience:** verified graceful fallback handling for unseen partners, unseen SKUs, missing serials, and unobserved categories (`test_model.py`).
6. **Local review service API:** verified `GET /api/health`, `POST /api/score`, missing-field validation errors, and informative employee review signals (`test_service.py`).
