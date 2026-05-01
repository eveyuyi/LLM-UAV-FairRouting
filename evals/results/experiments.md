# Experimental Results

---

## 5.1 Experimental Setup

**Scenario.** Experiments are conducted on a simulated urban UAV logistics network. The routing evaluation set comprises 96 five-minute dispatch windows (time slots 0–47, covering the first 4 hours of the simulated day), 722 delivery demands, and 4,340 demand records across 576 windows for extraction quality analysis. All experiments use seed 4111. Each dispatch window is solved **independently**: UAVs reset to their home station at the start of each window, so delivery latency reflects within-window service time only and is not inflated by cross-window queuing artefacts. *(Note: the current results in §5.3–5.5 were produced under the legacy continuous-simulation mode and will be re-run with independent windows.)*

**Compared methods.** Six configurations vary the priority assignment strategy while holding the NSGA-III multi-objective solver constant:

| ID | Priority Strategy | Description |
|:--|:--|:--|
| **M0a** | Random | Priority drawn uniformly at random from {1, 2, 3, 4}; serves as a no-information lower bound. |
| **M0b** | Uniform | All demands receive identical weight; equivalent to priority-agnostic scheduling. |
| **M0c** | Rule-based | Hand-crafted heuristics map cargo type, requester role, and deadline to a priority tier; represents the standard operational practice. |
| **M1\_pre** | LLM (zero-shot) | Qwen3-4B base model applied zero-shot as the priority ranker (LLM3), without task-specific fine-tuning. |
| **M1\_ft** | LLM (fine-tuned) | Qwen3-4B fine-tuned with supervised learning followed by GRPO reward optimization on human-aligned preference data. |
| **M1\_gemini** | LLM (frontier API) | Gemini-3.1-flash-lite applied zero-shot; provides an upper-reference bound from a frontier commercial model. |

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

### 5.3.1 Operational Metrics

| Metric | M0a | M0b | M0c | M1\_pre | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|
| **Overall service rate** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Overall on-time rate | 35.6% | 37.1% | 29.5% | 32.1% | 31.2% | 33.1% |
| Mean delivery latency (min) | 262.6 | 245.3 | 258.9 | 261.8 | 257.4 | 260.8 |
| P90 latency (min) | 583.5 | 562.4 | 578.1 | 575.2 | 580.6 | 585.8 |
| **P1 (critical) on-time rate** | 15.9% | 22.7% | 54.5% | 60.2% | **63.6%** | 67.0% |
| **P1 mean latency (min)** | 261.9 | 249.8 | 29.6 | 62.5 | **13.5** | 13.6 |
| Urgent (P1+P2) on-time rate | 22.3% | 23.9% | 56.9% | 57.5% | 62.2% | **67.0%** |
| Urgent mean latency (min) | 258.3 | 258.5 | 39.8 | 90.1 | **31.7** | 31.5 |
| **Priority-weighted on-time** | 30.4% | 32.2% | 40.9% | 42.3% | **44.6%** | 46.6% |
| P1–P4 latency gap (min) | 1.4 | −10.3 | 350.6 | 283.8 | **368.7** | 367.1 |
| Gini coefficient (latency) | 0.489 | 0.500 | 0.491 | 0.478 | 0.493 | 0.493 |
| Jain index (cross-tier) | 1.000 | 0.999 | 0.657 | 0.728 | 0.476 | 0.461 |
| Elderly recipients: on-time | 37.6% | 37.3% | 38.5% | 35.1% | **39.8%** | 37.3% |
| Elderly recipients: mean latency (min) | 244.7 | 251.1 | 213.5 | 240.5 | **210.2** | 224.2 |
| Vulnerable community: on-time | 36.5% | 37.1% | 32.5% | 34.6% | 34.4% | 34.0% |
| Total flight distance (km) | 121,095 | 116,125 | 120,636 | 118,663 | 121,124 | 122,906 |

### 5.3.2 Mean Delivery Latency by Priority Tier (min)

| Tier | M0a | M0b | M0c | M1\_pre | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|--:|--:|
| P1 — critical | 261.9 | 249.8 | 29.6 | 62.5 | **13.5** | 13.6 |
| P2 — urgent | 255.2 | 266.1 | 48.7 | 114.4 | 47.8 | **47.2** |
| P3 — standard | 268.2 | 246.0 | 112.6 | 199.7 | 106.6 | 143.0 |
| P4 — routine | 263.3 | 239.6 | 380.2 | 346.4 | 382.2 | 380.7 |

**Priority differentiation.** The random baseline (M0a) and uniform baseline (M0b) produce near-uniform latency across all four tiers (inter-tier range < 22 min), confirming that neither strategy achieves meaningful resource triage. The rule-based baseline (M0c) introduces substantial differentiation — a 350.6 min gap between P1 and P4 — by front-loading critical deliveries. The fine-tuned LLM (M1\_ft) extends this gap further to 368.7 min while simultaneously *reducing* P1 mean latency from 29.6 min (M0c) to 13.5 min, a 54% improvement over the strongest baseline. The zero-shot LLM (M1\_pre) achieves meaningful differentiation (283.8 min gap) but lags behind M0c in P1 latency, underscoring the value of preference-aligned fine-tuning.

**Priority-weighted on-time rate.** This metric weights on-time delivery by tier importance, providing a fairer measure of triage quality than the raw aggregate rate. M1\_ft achieves 44.6% versus M0c's 40.9% (+3.7 pp) and M0a's 30.4% (+14.2 pp). M1\_gemini reaches 46.6%, indicating that a fine-tuned 4B parameter model (M1\_ft) closely approaches frontier API performance at substantially lower deployment cost.

**Aggregate on-time rate.** M0c (29.5%) and M1\_ft (31.2%) fall below the non-discriminating baselines M0a/M0b (35–37%) in raw aggregate on-time rate. This reflects the intended trade-off: concentrating UAV capacity on critical demands necessarily delays routine deliveries. The priority-weighted metric rather than the aggregate on-time rate captures the system's allocation objective.

**Equity considerations.** The cross-tier Jain fairness index is low for M1\_ft (0.476) and M1\_gemini (0.461), as these systems explicitly allocate unequal service across priority tiers — a design goal, not a deficiency. The Gini coefficient over *individual* delivery latencies (0.493 for M1\_ft) matches that of M0a (0.489), indicating that prioritization does not increase within-tier dispersion. Service rates for elderly and vulnerable community recipients remain 100% under all methods; M1\_ft achieves the lowest mean latency for elderly recipients (210.2 min).

---

## 5.4 LLM3 Priority Alignment Analysis (Contribution 3)

Priority alignment is evaluated by comparing each model's assigned priority to the `extraction_observable_priority` ground-truth label — the priority derivable from the LLM2-extracted structured demand, without access to latent or dialogue-level information inaccessible to LLM3.

| Metric | M0c† | M1\_pre | M1\_ft | M1\_gemini |
|:--|--:|--:|--:|--:|
| Demand count | 722 | 661 | 722 | 722 |
| Accuracy | 1.000† | 0.457 | **0.619** | 0.489 |
| Macro-F1 | 1.000† | 0.402 | **0.620** | 0.473 |
| Weighted-F1 | 1.000† | 0.428 | **0.624** | 0.485 |
| P1 recall | 1.000† | 0.500 | 0.790 | **0.895** |
| P1 precision | 1.000† | 0.270 | **0.523** | 0.300 |
| P1 F1 | 1.000† | 0.351 | **0.629** | 0.449 |
| Urgent (P1+P2) recall | 1.000† | 0.634 | 0.851 | **0.900** |
| Urgent F1 | 1.000† | 0.719 | **0.892** | 0.855 |
| Pairwise accuracy | 1.000† | 0.745 | 0.844 | **0.863** |
| Spearman ρ | 1.000† | 0.589 | **0.770** | 0.725 |
| Kendall τ | 1.000† | 0.517 | **0.695** | 0.633 |

†The rule-based baseline (M0c) is evaluated against `extraction_observable_priority`, which is generated by the same rule set that produces M0c's priority weights. The reported score of 1.00 therefore reflects circular validation rather than genuine generalization ability, and is included as a reference boundary.

**Effect of fine-tuning.** Preference-aligned fine-tuning (M1\_ft vs M1\_pre) yields substantial improvements across all ranking metrics: pairwise accuracy increases from 0.745 to 0.844 (+10 pp), P1 recall from 0.500 to 0.790 (+29 pp), and urgent recall from 0.634 to 0.851 (+22 pp). Gains are most pronounced on the highest-priority tier (P1), which directly explains the operational improvements observed in §5.3 — most notably the 78% reduction in P1 mean latency (62.5 → 13.5 min).

**Frontier model profile.** Gemini (M1\_gemini) achieves the highest P1 recall (0.895) and pairwise accuracy (0.863), yet its classification accuracy (0.489) and P1 F1 (0.449) remain below M1\_ft. This apparent inconsistency stems from a high-recall, low-precision pattern: Gemini systematically over-predicts P1 (P1 precision = 0.300), classifying many P2/P3 demands as critical. The resulting over-prioritization inflates the urgent queue but ensures that genuinely critical demands are almost never missed, explaining why M1\_gemini achieves the best operational P1 on-time rate (67.0%) and urgent on-time rate (67.0%). M1\_ft, with its more balanced P1 precision–recall profile (0.523 / 0.790), is preferable in resource-constrained scenarios where false-positive upgrades carry opportunity cost.

**Hard evaluation.** To probe robustness beyond standard i.i.d. conditions, a hard evaluation set of 43 windows is constructed from `llm3_grpo_hard.jsonl`, comprising counterfactual priority reassignment (30 windows), surface-level contradiction resolution (6 windows), near-tie priority discrimination (6 windows), and mixed-priority scenarios (1 window). Phase-2 solver results are pending (SLURM job 15297351); standard vs. hard-set performance gap will be reported upon completion.

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

To illustrate where LLM-based priority assignment qualitatively differs from rule-based triage, we compare the 722 priority predictions of the rule-based baseline (M0c) against those of the fine-tuned model (M1\_ft), using ground-truth latent priority as reference.

The rule-based system produces **zero cases** of critical under-detection (no demand with ground-truth P1 or P2 is assigned P3/P4 by the rules). However, it over-prioritizes **41 non-urgent demands**, assigning P1 or P2 to demands whose ground-truth priority is P3 or P4; M1\_ft correctly assigns P3 or P4 to all 41. This asymmetry reflects the known limitation of rule-based triage: when surface-level signals are ambiguous, conservative heuristics default to high priority, inflating the urgent queue at the cost of delaying genuinely routine deliveries.

**Illustrative pattern.** In each of the five largest-gap cases (rule = P2, ground truth = P4, M1\_ft = P4), a consumer or office administrator requests routine OTC medication delivery with a 120-minute deadline. The rule system detects `cargo_type = medication` and `deadline ≤ 120 min`, triggering a P2 assignment. M1\_ft reads the full structured context — `requester_role = consumer/office administrator`, `cargo = over-the-counter medication`, no clinical indicators, no vulnerability signals — and correctly assigns P4. The rule conflates *medical cargo* with *medical urgency*; the fine-tuned LLM distinguishes contextual severity.

This finding highlights a core advantage of learned priority ranking: rather than firing on isolated field values, LLM3 jointly reasons over the full set of extracted signals to distinguish surface similarity from semantic equivalence.
