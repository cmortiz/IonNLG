# Trapped-ion raw data (G14 game)

Circuits are numbered 0-13 for the vertex questions and 14-50 for the edge questions, matching `circuits/line N (u, v).txt`, where `(u, v)` are the questions sent to player 0 and player 1.

| File | System | Date | Shots per circuit | Contents |
|---|---|---|---|---|
| `Blue data.txt` | Duke Blue | 2024-10 | 1939-2032 (varies by circuit) | outcome probabilities per circuit |
| `Gold data.json` | Duke Gold | 2024-11 | 300 | outcome probabilities per circuit |
| `silver_raw_probabilities.csv` | UMD Silver | 2024-07-18 | 2000 (circuits 0-27), 4000 (28-50) | 5-bit outcome probabilities (one spare ion, ignored in the analysis) |
| `silver_spam_corrected_probabilities.csv` | UMD Silver, SPAM-corrected | 2024-07-18 | as above | 4-bit outcome probabilities |
| `silver_unmitigated.csv`, `silver_mitigated.csv` | UMD Silver | 2024-07-18 | as above | per-circuit win rates used in the first preprint version; superseded by the two files above |
| `ionq_winrates.json` | IonQ Aria | 2024-11 | 20000 | per-circuit win rates |
| `ionq_forte_2026-03-26_counts.json` | IonQ Forte | 2026-03-26 | 2400 | outcome counts per circuit |
| `ionq_forte_2026-06-05_counts.json` | IonQ Forte | 2026-06 | 3100 | outcome counts per circuit |
| `quantinuum_h2_2026-06-15_counts.json` | Quantinuum H2 | 2026-06-15 | 3000 | outcome counts per circuit (list in circuit order) |

For a 4-bit outcome with integer value k, player 0's answer is k % 4 and player 1's answer is k // 4 (see `scripts/pbr_test.py`). In the Forte files, the circuit index is the numeric prefix of each result key.

The revised analysis computes the Silver win rate from `silver_raw_probabilities.csv`, giving 0.954(3); the per-circuit file `silver_unmitigated.csv` from the first preprint version gives 0.953 under the same weights. The March 2026 Forte jobs were submitted in IonQ's native gate set with debiasing disabled. The H2 jobs were submitted through Quantinuum's default compilation stack.
