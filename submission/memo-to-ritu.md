# Memo: Warranty Claim Review Pilot

**To:** Ritu Deshpande, Head of D2C Operations  
**Subject:** A 40-Claim-a-Month Fraud Review Queue for Kestrel Home

I built a warranty-claim fraud risk scoring and ranking service designed to prioritize the investigation desk's approximately 40-claim monthly review capacity. It provides human reviewers with risk scores and operational review signals; it is decision support to guide human review, not an automatic rejection system.

### Validation Findings & Economic Value

In our primary chronological validation (evaluating 2,363 claims submitted after 28 March 2026, containing 48 true fraud cases):
- **Model Review Queue (Top 40):** Captured **5 fraud cases** (12.5% precision@40, 10.42% recall@40), avoiding an estimated **₹45,338** in gross fraudulent payouts.
- **Net Economic Value:** After factoring in ₹13,300 in customer goodwill penalty for holding 35 genuine claims (₹380/claim) and ₹10,400 in review overhead (40 claims × ₹260), estimated net economic value was **₹21,638**.
- **Claim-Amount Benchmark:** Sorting by claim amount alone captured 4 frauds (10.0% precision, 8.33% recall) and produced **₹35,161** net value on this specific window, because those 4 claims happened to have larger individual payouts.
- **Random Selection Baseline:** An unguided 40-claim sample would be expected to capture only ~0.8 fraud cases (2.03% precision) with an expected net loss of approximately -₹23,037.

### Honest Multi-Period Perspective

Results vary across time. On the latest single window, the model captured incremental fraud (+1 case, a 25% gain in detection), but did not beat the claim-amount heuristic on net rupee recovery. Across five sequential historical chronological windows, however, the model averaged **₹71,038** in net value (and 12.0 fraud cases caught) compared to **₹47,899** (and 3.8 fraud cases caught) for the claim-amount heuristic.

Furthermore, a pure claim-amount heuristic is vulnerable to adversarial gaming: dishonest partners quickly learn monetary cutoffs and submit multiple smaller claims just below the threshold. The model evaluates multi-signal risk across all claim amounts.

### Operational Recommendation: Shadow-Mode Rollout

Because fraud patterns shift over time, I recommend a controlled shadow-mode pilot:
1. **Silent Parallel Execution:** Run the scoring service silently alongside current desk operations for one month without altering customer payouts.
2. **Reviewer Decision Support:** Provide investigation staff with the continuous risk score along with the operational review signals (e.g., partner onboarding history, list-price ratio, inspection sign-off gaps, customer claim counts, and uncataloged partner/SKU fallbacks).
3. **Monitor Realized Net Value:** Track actual fraud capture and net recovery each month. If realized performance falls below simple baselines, re-evaluate ranking criteria or suspend queue sorting.

To directly maximize rupees saved per claim checked, future queue sorting can rank claims by **Expected Loss Avoided** ($\text{Score} = P(\text{fraud}) \times \text{claim\_amount} - (1 - P(\text{fraud})) \times \text{goodwill\_cost}$), pairing risk likelihood with financial exposure.
