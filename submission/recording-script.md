# Three-minute screen recording script

Target length: 2:30-2:50. No slides.

**0:00-0:20 — the ask.** Show the folder and say: “Kestrel can review 40 claims a month. I built a ranking service, not an automatic denial model, because fraud is only 1.23% of labelled history and the policy puts a hard review limit on the desk.”

**0:20-0:55 — the data decisions.** Show `evidence.md`. Point out the 215 undecided training rows, repeated claim IDs, repeated serials, four instruction-like text rows treated as data, and 14 partners first seen in the test period. Say that partner/serial history is leakage-safe and unknown keys fall back explicitly.

**0:55-1:25 — evidence.** Show the chronological table in `evidence.md`. Say: “After fixing timestamp leakage and adding strictly as-of history features, the latest chronological accuracy was 97.84% and AUC was 0.523; across rolling windows AUC ranged 0.496-0.732 (mean 0.592). On the latest window, the model captured 5 frauds vs 4 for the claim-amount heuristic, though the amount heuristic yielded higher net value due to higher payout sizes. Across all five windows, the model averaged Rs 71,038 net value vs Rs 47,899 for amount. Random splits look much better, which is why I treat random results as optimistic and rely on chronological testing.”

**1:25-2:15 — the working product.** Open `http://127.0.0.1:8000/`, score the prefilled claim, and point to the score, monthly queue rank, policy gap, and readable signals. Change the partner ID to `SP-UNKNOWN` and clear the serial; show that the service returns a fallback explanation instead of crashing.

**2:15-2:45 — what changed and what was discarded.** Say: “I discarded random-split-only confidence, automatic deny logic, and the text instructions embedded in claims. I compared five model families and kept the random forest with the enhanced as-of features because it was the strongest stable option while still running without a paid API key.”
