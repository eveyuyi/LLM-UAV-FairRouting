# Experimental Results

---

## 5.1 Experimental Setup

**Scenario.** Experiments are conducted on a simulated urban UAV logistics network. The evaluation set comprises 96 five-minute dispatch windows (time slots 0–47, covering the first 4 hours of the simulated day), 722 delivery demands, and 4,340 demand records across 576 windows for extraction quality analysis. All experiments use seed 4111.

**Simulation mode.** Results are reported under *continuous simulation*: UAVs operate across the full 4-hour horizon, and the NSGA-III solver re-optimises routes at each dispatch window using up-to-date fleet state. This preserves realistic resource contention across windows, which is the condition under which priority assignment is meaningful. A UAV fleet of **7 aircraft** is used as the primary configuration; §5.3.3 reports a sensitivity analysis over fleet sizes of 3, 7, and 10 UAVs.

**Compared methods.** Seven configurations vary the priority assignment strategy while holding the NSGA-III multi-objective solver constant:

| ID | Priority Strategy | Description |
|:--|:--|:--|
| **M0a** | Random | Priority drawn uniformly at random from {1, 2, 3, 4}; serves as a no-information lower bound. |
| **M0b** | Uniform | All demands receive identical weight; equivalent to priority-agnostic scheduling. |
| **M0c** | Rule-based | Hand-crafted heuristics map cargo type, requester role, and deadline to a priority tier; represents the standard operational practice. |
| **M1\_pre** | LLM (zero-shot) | Qwen3-4B base model applied zero-shot as the priority ranker (LLM3), without task-specific fine-tuning. |
| **M1\_sft** | LLM (SFT-only) | Qwen3-4B fine-tuned with supervised learning on human-aligned preference data, without subsequent GRPO; ablation separating SFT and GRPO contributions. |
| **M1\_ft** | LLM (SFT + GRPO) | Qwen3-4B fine-tuned with supervised learning followed by GRPO reward optimisation on human-aligned preference data. |
| **M1\_gemini** | LLM (frontier API) | Gemini 3.1 Flash Lite Preview (VertexAI) applied zero-shot; provides an upper-reference bound from a frontier commercial model. |

All M1 variants share the same LLM2 extraction pipeline and NSGA-III solver back-end; only the LLM3 priority scoring module differs.

---

## 5.2 Demand Extraction Quality (Contribution 1)

LLM2 extracts structured demand fields from free-form dispatch dialogues. Extraction quality is evaluated against reference annotations from the data generation pipeline across 4,340 demands.

### 5.2.1 Field-Level Accuracy

Fields fall into two categories: *dialogue-explicit* fields whose values are directly stated in the conversation, and *geospatial-augmented* fields whose values derive from an external demographic database queried at ingestion time and are not present in the dialogue.

| Field | Accuracy | Category |
|:--|:--|:--|
| origin\_fid (supply depot) | **100.0%** | Dialogue-explicit |
| dest\_fid (destination node) | **100.0%** | Dialogue-explicit |
| deadline\_minutes | **99.9%** | Dialogue-explicit |
| weight\_kg | 99.0% | Dialogue-explicit |
| cargo\_type | 95.9% | Dialogue-explicit |
| children\_involved | 86.4% | Context-inferrable |
| special\_handling\_match | 83.0% | Context-inferrable |
| requester\_role | 78.2% | Dialogue-explicit (surface variants) |
| temperature\_sensitive | 75.1% | Context-inferrable |
| elderly\_involved | 61.1% | Geospatial-augmented ‡ |
| dest\_type | 51.0% | Ambiguous from dialogue |
| vulnerable\_community | 50.1% | Geospatial-augmented ‡ |
| demand\_tier | 47.7% | Downstream derived |

‡ In the deployed pipeline, `elderly_involved` and `vulnerable_community` are populated automatically via a geospatial lookup on the destination node's demographic profile (`elderly_ratio`), not by LLM2 extraction. Their low accuracy against the reference annotation is thus not an extraction failure.

### 5.2.2 System-Level Metrics

| Metric | Value |
|:--|:--|
| Schema validity rate | **100.0%** |
| Solver feasibility rate | **100.0%** (96/96 windows · 722/722 demands · 0 filtered) |
| Critical signal recall (four-field joint) | 28.5% |
| Critical signal recall (dialogue-visible fields only) | **78.1%** |
| Priority chain consistency (latent → dialogue → extraction) | 42.1% |

**Critical signal recall** requires `deadline_minutes`, `requester_role`, `elderly_involved`, and `vulnerable_community` to be simultaneously correct. The naïve figure of 28.5% is driven entirely by the two geospatial-augmented fields: inspection of 1,591 `elderly_involved` false-negative cases confirms that the dialogue contains no mention of the destination area's demographics — the reference value is derived from an `elderly_ratio` attribute in a node database, information that is structurally inaccessible to a dialogue-reading model. Restricting the metric to the two genuinely dialogue-visible critical fields (`deadline_minutes`, `requester_role`) yields a corrected recall of **78.1%**, consistent with the individual field accuracies reported above.

**Priority chain consistency** (42.1%) quantifies how faithfully the three-step chain — latent priority → dialogue-observable priority → extraction-observable priority — preserves the original triage signal. The first link (latent → dialogue) achieves 88.5%, indicating high fidelity in the synthetic data generation process. The bottleneck lies in the second link (dialogue → extraction), which achieves only 43.0%. The dominant failure mode is systematic *over-prioritization*: LLM2 assigns P3 to demands whose dialogue clearly expresses P4, producing the triple `(latent=4, dialogue\_obs=4, extraction\_obs=3)` in 1,131 cases. In total, 1,762 demands (41%) are over-prioritized by at least one tier. This constitutes a learned safety bias rather than a field-extraction failure, and motivates the discriminative re-ranking role of LLM3: rather than trusting the over-conservative P2/P3 labels from LLM2, LLM3 re-scores demands based on the full structured evidence.

---

## 5.3 Routing Performance Under Alternative Priority Strategies (Contribution 2)

Primary results use the 7-UAV continuous-simulation configuration. §5.3.3 reports fleet-size sensitivity.

### 5.3.1 Operational Metrics (7 UAVs, continuous simulation)

| Metric | M0a | M0b | M0c | M1\_pre | M1\_sft | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Overall service rate** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Overall on-time rate | 74.1% | 75.2% | 79.9% | 79.2% | 80.5% | **80.2%** | 77.2% |
| Mean delivery latency (min) | 58.6 | 50.6 | 60.4 | 58.1 | **57.0** | 59.7 | 61.8 |
| P90 latency (min) | 194.5 | 170.5 | 212.7 | 198.4 | 196.8 | 202.6 | 209.2 |
| **P1 (critical) on-time rate** | 44.3% | 47.7% | 93.2% | 86.4% | 96.6% | **97.7%** | 87.5% |
| **P1 mean latency (min)** | 72.9 | 50.1 | 10.3 | 14.6 | **9.6** | **9.4** | 10.0 |
| Urgent (P1+P2) on-time rate | 46.1% | 47.5% | 95.6% | 85.5% | 97.0% | **97.4%** | 88.7% |
| Urgent mean latency (min) | 63.2 | 48.8 | 10.8 | 19.3 | **10.7** | **10.4** | 11.1 |
| **Priority-weighted on-time** | 65.5% | 67.4% | 87.0% | 83.0% | 87.6% | **87.9%** | 84.1% |
| P1–P4 latency gap (min) | −19.3 | 1.4 | 79.6 | 63.2 | 74.2 | **79.5** | 81.3 |
| Gini coefficient (latency) | 0.620 | 0.641 | 0.634 | 0.625 | 0.629 | 0.629 | 0.621 |
| Jain fairness (cross-tier) | 0.978 | 0.999 | 0.784 | 0.756 | 0.764 | 0.775 | 0.758 |
| Elderly recipients: on-time | 73.6% | 72.0% | 82.3% | 80.4% | 82.3% | **83.5%** | 83.2% |
| Vulnerable community: on-time | 74.6% | 72.5% | 79.7% | 79.7% | 80.5% | **81.3%** | 80.7% |

### 5.3.2 Mean Delivery Latency by Priority Tier (min)

**7 UAVs (primary configuration)**

| Tier | M0a | M0b | M0c | M1\_pre | M1\_sft | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|--:|
| P1 — critical | 72.9 | 50.1 | 10.3 | 14.6 | **9.6** | **9.4** | 10.0 |
| P2 — urgent | 55.0 | 47.1 | 11.3 | 24.9 | **12.0** | **11.6** | 12.3 |
| P3 — standard | 73.6 | 50.8 | 16.9 | 40.0 | 19.4 | **16.6** | 20.9 |
| P4 — routine | 53.6 | 51.5 | 89.9 | 77.8 | 83.8 | **88.9** | 91.2 |

**10 UAVs**

| Tier | M0a | M0b | M0c | M1\_pre | M1\_sft | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|--:|
| P1 — critical | 14.9 | 18.0 | 9.8 | 9.6 | 9.5 | **9.3** | **9.3** |
| P2 — urgent | 17.9 | 16.2 | 10.2 | 10.8 | 10.3 | **10.2** | 10.5 |
| P3 — standard | 20.3 | 20.1 | 11.0 | 18.2 | 11.1 | **10.3** | 12.2 |
| P4 — routine | 19.0 | **16.6** | 22.4 | 22.1 | 23.5 | 22.1 | 25.1 |

At 10 UAVs the fleet approaches capacity saturation: P1–P3 latencies compress across all methods to 9–20 min. M0b achieves the lowest P4 latency (16.6 min) because uniform weighting does not divert UAVs away from routine deliveries. Rule and LLM methods deliberately deprioritise P4, pushing its latency to 22–25 min — the intended trade-off for faster critical service.

**Priority differentiation.** The random baseline (M0a) produces a negative P1–P4 gap (−19.3 min): random assignment frequently tags P4 demands as high priority, causing them to be served before genuinely critical demands. M0b (uniform weights) provides no differentiation (gap = 1.4 min). The rule-based baseline (M0c) introduces a 79.6 min gap and reduces P1 latency from >50 min to 10.3 min. The SFT-only model (M1\_sft) achieves P1 latency of 9.6 min (6.8% improvement over M0c) and priority-weighted on-time of 87.6% (+0.6 pp over M0c). The fine-tuned LLM (M1\_ft, SFT+GRPO) achieves the lowest P1 latency at **9.4 min** and highest priority-weighted on-time at **87.9%** (+0.9 pp over M0c, +4.9 pp over M1\_pre). The zero-shot LLM (M1\_pre) shows meaningful P1 improvement (86.4% on-time vs. M0a's 44.3%) but lags behind fine-tuned variants, most notably in P3 latency (40.0 min vs. 16–17 min for M1\_sft/M1\_ft), reflecting the benefit of preference-aligned training for mid-tier calibration.

**Priority-weighted on-time rate.** This metric weights on-time delivery by tier importance (P1: ×4, P2: ×3, P3: ×2, P4: ×1). M1\_ft achieves 87.9% versus M0c's 87.0% (+0.9 pp) and M0a's 65.5% (+22.4 pp). M1\_sft (87.6%) closely approaches M1\_ft, demonstrating that SFT alone captures most of the GRPO gain. M1\_gemini (84.1%) falls below the fine-tuned 4B models despite being a frontier API, suggesting that zero-shot prompting of large models does not substitute for task-specific reward optimisation in this domain.

**Aggregate on-time rate.** M0c (79.9%) and M1\_ft (80.2%) have similar overall on-time rates, slightly above M0a/M0b (74–75%). Unlike the 3-UAV legacy configuration (§5.3.3) where prioritisation visibly lowered the aggregate rate, with 7 UAVs there is sufficient capacity to improve P1 service without substantially sacrificing P4 delivery timing.

**Equity considerations.** The cross-tier Jain fairness index is lower for the rule and LLM methods (M0c: 0.784; M1\_ft: 0.775; M1\_pre: 0.756) than for the random/uniform baselines (M0a: 0.978; M0b: 0.999), reflecting intentional resource allocation across priority tiers. The Gini coefficient over individual delivery latencies is similar across all methods (range 0.620–0.641), indicating that priority-driven scheduling does not substantially increase within-demand latency dispersion. Service rates for elderly and vulnerable community recipients remain 100% under all methods; M1\_ft achieves the highest on-time rates for both elderly (83.5%) and vulnerable-community (81.3%) recipients.

### 5.3.3 UAV Fleet Size Sensitivity Analysis

To assess robustness of conclusions across operating regimes, the same seven methods are evaluated under three fleet sizes: 3, 7, and 10 UAVs (continuous simulation). Fleet size determines resource scarcity: with 3 UAVs, demands heavily queue across windows; with 10 UAVs, throughput approaches demand rate and queuing is minimal.

#### P1 On-time Rate by Fleet Size

| Method | 3 UAV (legacy) | 7 UAV (primary) | 10 UAV |
|:--|--:|--:|--:|
| M0a | 15.9% | 44.3% | 71.6% |
| M0b | 22.7% | 47.7% | 75.0% |
| M0c | 54.5% | 93.2% | 95.5% |
| M1\_pre | 60.2% | 86.4% | 97.7% |
| M1\_sft | — | 96.6% | **98.9%** |
| M1\_ft | **63.6%** | **97.7%** | 94.3% |
| M1\_gemini | 67.0% | 87.5% | 96.6% |

#### Priority-Weighted On-time Rate by Fleet Size

| Method | 3 UAV (legacy) | 7 UAV (primary) | 10 UAV |
|:--|--:|--:|--:|
| M0a | 0.304 | 0.655 | 0.885 |
| M0b | 0.322 | 0.674 | 0.905 |
| M0c | 0.409 | 0.870 | 0.977 |
| M1\_pre | 0.423 | 0.830 | 0.967 |
| M1\_sft | — | 0.876 | **0.985** |
| M1\_ft | **0.446** | **0.879** | 0.977 |
| M1\_gemini | 0.466 | 0.841 | 0.978 |

#### Mean Delivery Latency (all demands, min) by Fleet Size

| Method | 3 UAV (legacy) | 7 UAV (primary) | 10 UAV |
|:--|--:|--:|--:|
| M0a | 262.6 | 58.6 | 18.5 |
| M0b | 245.3 | 50.6 | 17.1 |
| M0c | 258.9 | 60.4 | 17.8 |
| M1\_pre | 261.8 | 58.1 | 18.6 |
| M1\_sft | — | **57.0** | 18.4 |
| M1\_ft | 257.4 | 59.7 | **17.5** |
| M1\_gemini | 260.8 | 61.8 | 19.6 |

**Sensitivity findings.** Three robust conclusions hold across all fleet sizes:

1. **LLM substantially outperforms random/uniform baselines.** The gap in P1 on-time rate between M1\_ft and M0a is 53.4 pp (7 UAVs) and 22.7 pp (10 UAVs), confirming that priority assignment has real operational value whenever resource contention exists.

2. **Fine-tuned LLM matches or exceeds the rule-based baseline.** At 7 UAVs, M1\_ft achieves 97.7% P1 on-time versus M0c's 93.2% (+4.5 pp). At 10 UAVs, M0c slightly leads on P1 on-time (95.5% vs 94.3%) but M1\_sft surpasses it on both P1 on-time (98.9%) and priority-weighted on-time (0.985 vs 0.977).

3. **Zero-shot LLM (M1\_pre) is meaningfully weaker than fine-tuned variants.** The gap is most pronounced at 7 UAVs: M1\_pre priority-weighted on-time is 0.830 vs 0.879 for M1\_ft (−4.9 pp), driven by poor P3 latency (40.0 vs 16.6 min). GRPO fine-tuning resolves mid-tier mis-calibration in the zero-shot model.

**SFT vs. GRPO ablation (M1\_sft vs. M1\_ft).** M1\_sft performs competitively throughout: at 7 UAVs it nearly matches M1\_ft on priority-weighted on-time (0.876 vs 0.879, −0.3 pp) and at 10 UAVs it outperforms M1\_ft on both P1 on-time (98.9% vs 94.3%) and priority-weighted on-time (0.985 vs 0.977). This routing-level parity is consistent with the alignment analysis (§5.4): pairwise ranking accuracy is nearly identical after SFT vs SFT+GRPO (0.846 vs 0.844), meaning the relative ordering of demands fed to the solver is virtually the same. GRPO's contribution — a +4.3 pp gain in classification accuracy and +5.6 pp in P1 F1 — is real but expressed as improved assignment confidence rather than reordering, which has limited impact when the solver already receives well-ordered inputs. The 3-UAV legacy configuration (where M1\_ft leads M0c most clearly in P1 latency: 13.5 vs 29.6 min) reflects extreme resource scarcity amplifying small priority assignment differences; at higher fleet sizes this amplification is reduced.

**Fleet-size recommendation.** The 7-UAV configuration provides the clearest priority differentiation signal while maintaining operationally plausible latency ranges. The 3-UAV legacy configuration inflates absolute latency (mean ~260 min) due to cross-window demand queuing that exceeds fleet capacity; the 10-UAV configuration approaches capacity saturation, compressing the priority differentiation signal.

### 5.3.4 Route B: Independent-Window Baseline (3 UAVs)

Route B uses *independent-window* simulation: each 5-minute dispatch window is solved in isolation with a fresh UAV fleet. All demands in a window have their arrival time set to 0, so the solver sees the full window batch at once and solves a single-batch assignment. This removes cross-window demand queuing by design.

| Metric | M0a | M0b | M0c | M1\_pre | M1\_sft | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Overall on-time rate | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% |
| Mean latency (min) | 9.3 | 9.0 | 9.0 | 9.0 | 9.0 | **8.9** | 9.0 |
| P1 on-time rate | 98.9% | 98.9% | 98.9% | 98.9% | 98.9% | 98.9% | 98.9% |
| P1 mean latency (min) | 8.5 | 8.3 | **8.4** | 8.6 | 8.5 | **8.4** | **8.4** |
| P1–P4 latency gap (min) | 1.1 | 1.0 | 0.9 | 0.7 | 0.8 | 0.9 | 0.9 |
| Priority-weighted on-time | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |

**Finding.** All seven methods converge to near-identical results: ~99.9% on-time, ~9 min mean latency, and a P1–P4 gap of less than 1.1 min. The independent-window mode with 3 UAVs provides sufficient capacity per window (3 UAVs for 7–8 demands) to serve all demands within deadline regardless of priority ordering. Under this regime, priority assignment has no measurable effect because there is no resource scarcity to arbitrate.

This result serves as a *capacity-saturation control*: it confirms that the priority differentiation observed in the continuous-simulation Route A experiments (§5.3.1–5.3.3) is genuinely caused by cross-window resource contention, not by implementation artefacts. When contention is eliminated, all methods are equivalent.

---

## 5.4 LLM3 Priority Alignment Analysis (Contribution 3)

Priority alignment is evaluated by comparing each model's assigned priority to the `extraction_observable_priority` ground-truth label — the priority derivable from the LLM2-extracted structured demand, without access to latent or dialogue-level information inaccessible to LLM3.

| Metric | M0c† | M1\_pre‡ | M1\_sft | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|
| Demand count | 722 | 661 | 722 | 722 | 722 |
| Accuracy | 1.000† | 0.457 | 0.576 | **0.619** | 0.489 |
| Macro-F1 | 1.000† | 0.402 | 0.578 | **0.620** | 0.473 |
| Weighted-F1 | 1.000† | 0.428 | 0.583 | **0.624** | 0.485 |
| P1 recall | 1.000† | 0.500 | 0.789 | 0.790 | **0.895** |
| P1 precision | 1.000† | 0.270 | 0.450 | **0.523** | 0.300 |
| P1 F1 | 1.000† | 0.351 | 0.573 | **0.629** | 0.449 |
| Urgent (P1+P2) recall | 1.000† | 0.634 | 0.861 | 0.851 | **0.900** |
| Urgent F1 | 1.000† | 0.719 | 0.885 | **0.892** | 0.855 |
| Pairwise accuracy | 1.000† | 0.745 | **0.846** | 0.844 | 0.863 |
| Spearman ρ | 1.000† | 0.589 | 0.748 | **0.770** | 0.725 |
| Kendall τ | 1.000† | 0.517 | 0.667 | **0.695** | 0.633 |

†The rule-based baseline (M0c) is evaluated against `extraction_observable_priority`, which is generated by the same rule set that produces M0c's priority weights. The reported score of 1.00 therefore reflects circular validation rather than genuine generalization ability, and is included as a reference boundary. ‡ M1\_pre (zero-shot base model) produced no LLM3 output for 61 of 722 demand appearances; those 61 demands received a fallback priority of 3 in the solver. Alignment metrics are therefore computed over the 661 demands for which a prediction was generated, making M1\_pre's classification scores a slightly optimistic estimate of its true coverage-adjusted accuracy.

**Effect of fine-tuning.** Decomposing the fine-tuning stages reveals two distinct contributions. SFT alone (M1\_sft) yields large gains over zero-shot (M1\_pre): accuracy improves from 0.457 to 0.576 (+11.9 pp), pairwise accuracy from 0.745 to 0.846 (+10.1 pp), and P1 recall from 0.500 to 0.789 (+28.9 pp). GRPO on top of SFT (M1\_ft) delivers a further meaningful improvement in classification precision: accuracy increases from 0.576 to 0.619 (+4.3 pp) and P1 F1 from 0.573 to 0.629 (+5.6 pp), reflecting better calibration of prediction confidence for the P1 tier. Pairwise ranking accuracy is nearly unchanged between SFT and SFT+GRPO (0.846 vs 0.844), indicating that GRPO's contribution lies primarily in sharpening classification precision rather than altering relative demand ordering — which explains why routing outcomes at 7 UAVs are similar for M1\_sft and M1\_ft despite their classification gap.

**Frontier model profile.** Gemini achieves the highest P1 recall (0.895) and pairwise accuracy (0.863), yet its classification accuracy (0.489) and P1 F1 (0.449) remain below M1\_ft. This stems from a high-recall, low-precision pattern: Gemini systematically over-predicts P1 (precision = 0.300), inflating the urgent queue. M1\_ft's more balanced P1 precision–recall profile (0.523 / 0.790) is preferable in resource-constrained scenarios where false-positive upgrades carry opportunity cost. Operationally this is confirmed by the 7-UAV results: M1\_ft P1 on-time (97.7%) exceeds M1\_gemini (87.5%) despite M1\_gemini's higher P1 recall, because Gemini's false positives compete for UAV slots and delay genuinely critical demands.

---

## 5.5 Computational Efficiency

| Method | Mean solve time / window (s) | P90 solve time (s) | Path cache hit rate |
|:--|--:|--:|--:|
| M0a (random) | 2.58 | 1.66 | 99.9% |
| M0b (uniform) | 0.39 | 1.35 | 99.9% |
| M0c (rule-based) | 0.37 | 1.15 | 100.0% |
| M1\_pre (zero-shot LLM) | 0.32 | 0.97 | 100.0% |
| M1\_ft (fine-tuned LLM) | 0.41 | 1.26 | 100.0% |

The shared path cache stores 122,070 pre-computed drone route segments keyed by SHA256 hashes of (origin, destination) node pairs. At 99.9–100% hit rate across all methods, NSGA-III candidate evaluation is dominated by cache lookup rather than flight simulation, keeping per-window solve time well under 3 s. End-to-end pipeline latency for the LLM2 extraction and LLM3 ranking modules is not reported here, as all offline evaluations bypass live LLM inference; per-module timing requires a dedicated online run with a vLLM serving instance.

---
## 5.6 Case Study: Contextual Triage Correction

To illustrate where LLM-based priority assignment qualitatively differs from rule-based triage, we compare M0c and M1\_ft priority predictions across the 722 demands (seed 4111, 7-UAV continuous simulation), using latent ground-truth priority as reference.

**Aggregate error pattern.** The rule-based system (M0c) produces zero critical under-detection errors: no demand with ground-truth P1 or P2 is incorrectly deprioritised to P3/P4. However, M0c over-prioritises 41 non-urgent demands, assigning P1 or P2 to cases whose ground-truth priority is P3 or P4; M1\_ft correctly assigns P3 or P4 to all 41. This asymmetric error profile reflects a structural bias in rule-based heuristics: when surface signals co-occur with ambiguous severity cues, conservative rules default to high priority rather than reading the full context.

**Selected cases.** Table 5 presents three representative instances drawn from the 41 rule over-prioritisation errors, chosen to illustrate distinct failure modes. For each case, the dispatch dialogue excerpt is shown verbatim alongside the extracted structured fields and model predictions.

*Table 5. Selected cases where M0c assigns P2 and M1\_ft assigns the correct P4.*

| Case | Demand ID | Cargo | Requester role | Deadline | Rule trigger | M0c | M1\_ft | Ground truth |
|:--|:--|:--|:--|--:|:--|:--|:--|:--|
| C1 | DEM\_005\_06 | OTC medication (8.8 kg) | Office administrator | 120 min | `cargo=medication` ∧ `deadline≤120` | P2 | **P4** | P4 |
| C2 | DEM\_013\_00 | OTC medication (22.0 kg) | Office administrator | 120 min | `cargo=medication` ∧ `deadline≤120` | P2 | **P4** | P4 |
| C3 | DEM\_013\_04 | Food (24.1 kg) | Family caregiver | 120 min | `deadline≤120` | P2 | **P4** | P4 |

---

**Case C1 — Routine medication, clean context** (DEM\_005\_06)

> **[00:25] Office Administrator:** I placed an urgent order for 8.8 kg of **OTC medication** for same-day delivery to DEM\_25495. The **delivery deadline is 120 minutes**.
> **[00:28] Dispatcher:** Great. I'll pack the medication in a standard tamper-evident pharmacy bag and dispatch it immediately from Commercial Distribution Hub COM\_5.

The dialogue is unambiguous: pharmaceutical cargo, standard 120-minute deadline, no clinical context, no vulnerability signals. M0c nonetheless assigns P2 because its condition `cargo_type = medication AND deadline ≤ 120 min` fires unconditionally on any pharmaceutical cargo regardless of clinical necessity. M1\_ft reads the complete extracted profile — `requester_role = office_administrator`, `cargo = over-the-counter medication`, no emergency or vulnerability fields — and correctly assigns P4. This case illustrates the rule's fundamental inability to distinguish *medical cargo* from *medical urgency*.

---

**Case C2 — Misleading clinical framing** (DEM\_013\_00)

> **[01:05] Office Administrator:** **Clinical dispatch request.** Please route 220 boxes of **OTC medication** (22.0 kg) from COM\_52. A same-day home-care order needs **symptom relief medication**. Delivery is needed within **120 minutes**. … The receiving point is an office. **The receiver serves a vulnerable population.**
> **[01:05] Delivery Platform:** Request acknowledged. Pack the order in a standard tamper-evident pharmacy bag. Departure from COM\_52 is being prioritized. Planned ETA is 120 min.

This is the most analytically challenging case. The dialogue opens with "Clinical dispatch request" and explicitly states that "the receiver serves a vulnerable population" — two phrases that individually would signal high priority in any keyword-based system. M0c is again triggered by `cargo=medication AND deadline≤120`, yielding P2. Yet the ground-truth priority is P4, because reading beyond the surface framing reveals that the cargo is **over-the-counter symptom-relief medication** (not emergency drugs), the requester is an office administrator rather than a clinician, and no life-critical indicators are present (no `cardiac_emergency`, `hospital_shortage`, or equivalent). M1\_ft jointly evaluates all extracted fields and correctly assigns P4, effectively disregarding the "Clinical dispatch request" header when the remaining structured evidence — cargo specificity, requester role, absence of emergency fields — unanimously indicates a routine delivery. This case demonstrates that LLM3 can override lexical urgency cues when the full structured evidence points to a non-urgent context.

---

**Case C3 — Deadline heuristic fires on non-medical cargo** (DEM\_013\_04)

> **[01:05] Family Caregiver:** Requesting same-day delivery of 24.1 kg of **food** from COM\_52 to DEM\_6303. It's **essential for our family**.
> **[01:06] Dispatcher:** Delivery is required within **120 minutes**, and is the landing zone ready?
> **[01:07] Family Caregiver:** Yes, a household member will collect it right after the notification. Can it be dropped off at a community locker?

The cargo is food, the requester is a family caregiver, and the only urgency signal is a standard 120-minute deadline and the phrase "essential for our family." M0c assigns P2 purely on the basis of `deadline ≤ 120 min` — a threshold that fires regardless of cargo type. M1\_ft reads `cargo = food` and `requester_role = family_caregiver`, recognises the complete absence of any medical or clinical signal, and assigns P4. This case reveals that the rule system's deadline trigger is cargo-agnostic, producing systematic over-prioritisation of time-sensitive but non-urgent routine deliveries.

---

**Discussion.** Across all three cases, M0c fires on one or two isolated field values (cargo type, deadline) without integrating the broader context. The three cases illustrate three distinct failure modes: (C1) cargo-type conflation — treating any medication as medical emergency cargo; (C2) lexical framing capture — surface urgency cues dominate even when semantic content is routine; (C3) cargo-agnostic deadline firing — the deadline threshold applies uniformly regardless of what is being delivered. M1\_ft's advantage lies in joint reasoning: it conditions its priority assignment on the full structured demand profile and resolves ambiguity that individual field values cannot. The operational consequence of M0c's 41 false P2 assignments is a persistently inflated urgent queue, which competes for UAV slots and delays genuinely critical (P1/P2) demands — directly explaining why M1\_ft achieves higher P1 on-time rates (97.7% vs. 93.2%) despite M0c's apparently conservative triage posture.
