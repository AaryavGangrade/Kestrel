# Task 2 V3 submission form

**What did you build, and what business outcome does it move? State the number and the money.**

I built `predictions.csv` plus a local one-claim scoring service. It ranks claims for the investigation desk's 40-claim monthly capacity and returns employee-readable relevant signals. In the primary chronological holdout, the top 40 contained 5 fraud claims (12.5% precision, 10.42% recall) and avoided Rs 45,338 gross payout; after Rs 380 goodwill per genuine hold and Rs 260 per review, estimated net value was Rs 21,638. On this same latest window, a simple highest-claim-amount heuristic captured 4 frauds (10.0% precision) and delivered Rs 35,161 net value. Therefore, the model demonstrates incremental fraud capture over the simple amount heuristic (+1 fraud case, +25% capture), but has not yet demonstrated superior economic value on this single window. Across five historical validation windows, however, the model averaged Rs 71,038 net value (ranging from Rs 21,638 to Rs 123,950) versus Rs 47,899 for the amount heuristic; this is an empirical historical range, not a monthly profit forecast.

**What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.**

I expect hidden ROC-AUC around 0.50-0.65 if hidden outcomes follow the observed chronological drift, with 97-99% accuracy at a fixed 0.5 threshold because fraud is rare (1.23% base rate). ROC-AUC is useful for comparing continuous ranking, but the operational business metric is top-40 queue capture. On the latest holdout, random selection would capture 0.81 frauds on average (2.03% precision, 1.69% recall; expected net loss Rs -23,037), while the model captured 5/48 (12.5% precision, 10.42% recall; Rs 21,638 net value) and the claim-amount heuristic captured 4/48 (10.0% precision, 8.33% recall; Rs 35,161 net value). ROC-AUC is not a probability estimate. I compared logistic regression, Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting on chronological windows after fixing timestamp leakage. Random Forest with enhanced as-of features was selected for ranking stability, multi-signal coverage, and clean local execution. The exact authoritative rerun in `evidence.json` reported primary holdout AUC 0.523, five-window mean AUC 0.592 (five-window range: 0.496-0.732; latest window 0.496), and accuracy 97.84-98.48% (mean 98.25%). Repeated random 80/20 splits gave AUC 0.790-0.941 (mean 0.879), which I treat as optimistic evidence only.

**How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.**

The primary check has 2,363 later labelled claims, with 48 frauds. Accuracy was 97.84%; at a 0.5 cutoff the model found 5 of 48 frauds and produced 8 false positives. The product uses ranking/top-40 review because that is the operational constraint. The latest top-40 precision was 12.5% (5/40) and recall@40 was 10.42% (5/48); rolling top-40 precision ranged 12.5-47.5% (mean 30.0%). Historical top-40 net value ranged from Rs 21,638 to Rs 123,950 (mean Rs 71,038). On the latest window, the model captured incremental fraud over the amount heuristic (5 vs 4) but achieved lower net value (Rs 21,638 vs Rs 35,161) because the amount heuristic's 4 frauds had larger payouts. Across all five windows, the model averaged 12.0 frauds caught vs 3.8 for the amount heuristic. It gets wrong cases from shifting partner cohorts and rare partners with little history. Earlier labelled history had 1.03% fraud prevalence, versus 2.03% in the latest window; mean claim amount fell from Rs 2,691 to Rs 2,321 and unique partners rose from 349 to 365. I tested prediction shape, duplicate IDs, API health, ordinary scoring, unknown partner/SKU and missing serial fallback, incomplete requests, and batch/online score parity.

**Did you change, narrow, or push back on the client's ask? What, when, and why?**

Yes. I narrowed “flag fraud” to “prioritise the 40 claims the desk can review” and rejected automatic denial. The API returns a prioritization score and requires monthly batch ranking rather than emitting a misleading fixed review boolean. A score of 0.30 does not mean a 30% probability of fraud. I made this decision after reading the policy and email thread: genuine holds cost Rs 380 goodwill, the desk has capacity for 40, and accuracy is misleading at 1.23% prevalence. I also did not assume newer partners are fraudulent; partner age is only one signal.

**What is wrong with what you are handing us, or with the data we handed you?**

Training includes 215 blank investigation outcomes; they are excluded rather than silently treated as genuine. Claim IDs repeat because partners resubmit; serials are messy and repeat across train/test. Four claim descriptions contain instruction-like text; they are untrusted data and not used as instructions. Fourteen test-period partners have no labelled history and use a fixed 1% smoothing prior. Target-history rates, counts, and trend features now use only labelled rows strictly earlier than each claim timestamp. Batch and one-record scores were checked and matched within 0.000001. Chronological AUC is unstable (0.496-0.732, latest repeated window 0.496); the model is not safe for autonomous payout decisions.

**What did you deliberately leave out, and why that rather than something else?**

I left out an LLM, paid APIs, automatic rejection, and free-text instruction following. A paid model would add unavailable operational cost and reproducibility risk; the Random Forest is easier to audit. I tested recency-weighted partner history but discarded it because its apparent gain was not stable in the final end-to-end rerun. I also did not turn `claim_id` into a memorised label because repeated IDs are resubmissions, not independent evidence.

**Anything you built or found that nobody asked for?**

The service returns explicit employee reasons and a missing-key fallback. I also produced repeated chronological and random split evidence, a money calculation using the policy costs, and a short recording script.

**What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.**

AI assistance was used to inspect the pack, draft and debug the deterministic Python model/service, and review the submission wording. No model API or paid inference call is used inside the product; incremental prediction cost is Rs 0. I discarded prompt-injection text in the data, random-split-only confidence, and automatic-denial logic. **Screen recording:** supplied separately by the candidate.

**Your Public Google Drive Link**

The screen recording is supplied separately by the candidate.

**Someone picks this up on Monday and you are unreachable. The three things they need to know.**

1. Run `python src/model.py` to regenerate predictions/evidence, then `python src/service.py` and open `http://127.0.0.1:8000/`. You can also execute the automated test suite with `pytest tests/`.
2. Use the score only to rank the 40 monthly investigations; never auto-deny from it. Watch rolling top-40 precision and net value.
3. Blank outcomes are undecided, unknown partner/SKU keys use fallback rates, and the chronological AUC drift warning is real.

**Honest hours spent.**

4

**What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)? Show the arithmetic.**

No paid calls. One prediction uses local deterministic Python only: Rs 0 per claim × 750 claims/month = Rs 0/month in API cost. Human review remains separately capacity-limited at 40 claims/month; the model does not claim that review labour is free.
