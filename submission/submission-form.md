# Task 2 V3 submission form

**What did you build, and what business outcome does it move? State the number and the money.**

I built a warranty-claim fraud risk scoring service and predictions.csv. The service ranks claims so the investigation desk can focus on its approximately 40-claim monthly review capacity and provides employee-readable review signals.

On the primary chronological holdout, the top 40 claims contained 5 fraud cases, giving 12.5% precision@40 and 10.42% recall@40. The estimated gross payout avoided was ₹45,338; after ₹13,300 estimated goodwill cost for 35 genuine holds and ₹10,400 for 40 reviews, estimated net value was ₹21,638.

Across five historical chronological windows, estimated net value averaged approximately ₹71,038. These are historical validation results, not forecasts.

**What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.**

I expect hidden ROC-AUC around 0.50–0.65, with substantial uncertainty because performance varies across time. Across five chronological windows, AUC ranged from approximately 0.496 to 0.732, with a mean around 0.592; the primary chronological holdout was approximately 0.523.

I did not repeatedly optimize against these same windows solely to increase AUC, because that could overfit the validation periods. ROC-AUC is used for the continuous risk ranking, while precision@40 is the operational metric for the approximately 40-claim review workflow. Random splits were materially more optimistic and were therefore not treated as the main estimate of future performance.

**How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.**

On the primary chronological holdout there were 2,363 later-labelled claims and 48 fraud cases. At the default 0.5 threshold, accuracy was 97.84%, with 5 true positives, 8 false positives, 2,307 true negatives and 43 false negatives.

Because the operational workflow is a ranked review queue rather than automatic rejection, I also evaluated the top 40 claims: 5 frauds were found, giving 12.5% precision@40 and 10.42% recall@40. Random selection would be expected to find only about 0.8 fraud cases in 40 claims.

I also tested multiple chronological windows, unseen partners/SKUs, missing serials, duplicate IDs, batch-vs-single scoring consistency, API health, schema validity and prediction generation.

The main limitation is temporal variation: fraud behaviour and partner patterns change over time, and individual chronological windows contain relatively few fraud cases.

**Did you change, narrow, or push back on the client's ask? What, when, and why?**

Yes. I changed the framing from automatic fraud flagging/rejection to ranking claims for a constrained human review queue. Accuracy alone is not sufficient for this workflow because fraud is rare and a high accuracy can coexist with many missed fraud cases.

The final system produces a continuous risk score and employee-readable review signals rather than automatically denying claims. I also did not assume that newer partners are fraudulent; partner age is only one feature and the data is allowed to determine its effect.

**What is wrong with what you are handing us, or with the data we handed you?**

215 claims with blank investigation outcomes were excluded from supervised training.

Claim IDs and serials can repeat, so repeated records are not treated as independent evidence. Test-period partners/SKUs without labelled history use fallback/smoothing behaviour. Missing serials are handled explicitly.

Some claim descriptions contain instruction-like text; these are treated as untrusted data rather than instructions to the model.

Historical target/rate features are constructed strictly as-of the claim timestamp, so future labels are not available to the claim being scored.

The main remaining limitation is temporal variation in model performance.

**What did you deliberately leave out, and why that rather than something else?**

I deliberately left out an LLM, paid inference APIs, automatic warranty rejection, and free-text instruction following.

The final prediction pipeline is deterministic/local, which avoids per-claim API cost and makes the scoring process easier to reproduce and audit.

I did not use claim_id as a memorised fraud label; repeated IDs are treated as repeated/resubmitted claims rather than independent evidence.

I retained the strictly-as-of 180-day recency-weighted partner history because it is leakage-safe and is part of the final validated feature set.

**Anything you built or found that nobody asked for?**

I added employee-readable review signals and explicit fallback handling for unseen partners/SKUs and missing serials.

I also evaluated multiple chronological windows, compared against random selection and a claim-amount heuristic, and included rupee-based evaluation using the policy costs.

The final system also checks batch-vs-single scoring consistency and provides a simple service interface for one-record review.

**What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.**

I used ChatGPT for code review, debugging, feature ideas, validation review and submission documentation. I did not use an LLM inside the final prediction pipeline.

AI assistance was particularly useful for reviewing leakage risks, validation design and implementation issues. I considered/discussed LLM-based claim interpretation but did not include it in the final pipeline because it added complexity and cost without being necessary for the final workflow.

The final prediction service runs locally using the Random Forest model and does not make paid AI API calls.

**Your Public Google Drive Link**

https://drive.google.com/file/d/1kPx-3K6D8KZ8rjV1kfjKprVydFNYWR__/view?usp=sharing

**Someone picks this up on Monday and you are unreachable. The three things they need to know.**

Run: python src/model.py to generate the predictions, then python src/service.py and open http://127.0.0.1:8000/. Run pytest tests/ to verify the system.

Use the score to rank claims for review. Focus the approximately 40 monthly reviews on the highest-risk claims rather than treating the score as a probability or automatically rejecting claims.

The key evidence is in evidence.md. It contains the chronological validation results, top-40 performance, business-value calculation, model-selection rationale, and comparisons against simpler baselines.

**Honest hours spent.**

7

**What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)? Show the arithmetic.**

The final prediction pipeline uses a local Random Forest and no paid inference API.

Paid inference cost per prediction: ₹0.

At 750 claims/month, paid inference cost would therefore be ₹0/month.
