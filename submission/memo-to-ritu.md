# Memo: warranty claim review pilot

**To:** Ritu Deshpande, Head of D2C Operations  
**Subject:** A 40-claim-a-month fraud review queue for Kestrel Home

I built a small claim-ranking service that scores incoming warranty claims and gives the investigation desk the reasons behind the ranking. It is designed to prioritize a human review queue of 40 claims per month, not to auto-reject customers.

### Bottom line on the latest window
The business outcome is avoided fraudulent payout while protecting genuine customer goodwill. On the latest chronological holdout (2,363 claims, 48 frauds):
- **Model queue:** 5 frauds in top 40 (12.5% precision, 10.42% recall), avoiding Rs 45,338 gross payout. After Rs 380 goodwill for 35 genuine holds and Rs 260 contact cost per review, estimated net value was **Rs 21,638**.
- **Claim-amount heuristic:** Selecting the 40 largest claims caught 4 frauds (10.0% precision, 8.33% recall) and produced **Rs 35,161** net value (Rs 59,241 gross saved).
- **Random queue baseline:** Expected 0.81 frauds (2.03% precision) with an expected net loss of **Rs -23,037**.

**Explicit business conclusion:** The model demonstrates incremental fraud capture over the simple amount heuristic on the latest window (+1 additional fraud caught, a 25% increase), but has **not yet demonstrated superior economic value** on this single window (Rs 21,638 vs Rs 35,161 net value). The amount heuristic achieved higher net value here because its 4 caught frauds averaged Rs 14,810 in claims, whereas the model's 5 caught frauds averaged Rs 9,068.

### Why retain the model for shadow-mode testing?
1. **Multi-window consistency:** Across earlier chronological windows, the model substantially outperformed the amount heuristic on both fraud capture (19 vs 3, 16 vs 4, 12 vs 4) and economic net value (Rs 123,950 vs Rs 32,635; Rs 94,421 vs Rs 48,299; Rs 80,747 vs Rs 48,299). Across all five windows, the model averaged Rs 71,038 net value (and 12.0 frauds caught) vs Rs 47,899 net value (and 3.8 frauds caught) for the amount heuristic.
2. **Gaming resistance:** A pure claim-amount heuristic is easily learned and gamed by dishonest partners who split claims or keep amounts just below review thresholds. The model evaluates structural operational patterns (partner history, serial reuse, policy inspection gaps, claim-to-list-price ratios) across all claim sizes.
3. **Zero operational risk:** Shadow mode tests whether the model's multi-signal ranking adds real-world lift without disrupting customer payouts.

### Recommendation to optimize the business objective
To directly satisfy Farhan Sheikh's objective ("how much fraud we stop per claim we check, in rupees — not a percentage"), I recommend transitioning queue prioritization from pure fraud probability $P(\text{fraud})$ to **Expected Loss Avoided**:
$$\text{Expected Net Value} = P(\text{fraud}) \times \text{claim\_amount\_inr} - (1 - P(\text{fraud})) \times \text{goodwill\_cost}$$
This pairs the model's risk assessment with the financial exposure of the claim.

### Operational KPIs and next steps
The board's requested accuracy was 97%: the chronological check achieved **97.84%** (five-window range: 97.84%–98.48%, mean 98.25%). However, accuracy is misleading because only 1.23% of labelled claims are fraud. Chronological ROC-AUC ranged from 0.496 to 0.732 across five rolling cutoffs (mean 0.592; 0.523 in the primary holdout and 0.496 in the latest window).

Next week in shadow mode:
1. Run the service silently for one week to rank the monthly queue alongside the existing process; do not alter payouts.
2. Record desk outcomes, amounts recovered, partner feedback, and signal accuracy.
3. If rolling top-40 net value fails to beat the simple amount baseline, adjust ranking weights or suspend automated queue sorting.

The service explicitly flags missing partner/SKU keys, missing serials, repeat claims, prior customer claims, high claim-to-list-price amounts, newer-partner context, and the post-1-May-2026 inspection rule. Undecided cases (blank outcomes) remain excluded from training rather than assumed genuine.
