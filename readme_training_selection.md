# Selection Training Overview

## Goal

This training pipeline teaches an LLM to act as a **policy-aware selector** over an already computed Pareto frontier.

The model is **not** solving UAV routing directly. Instead, it receives a structured description of:

- the current scene and demand set
- several Pareto candidate groups / solutions
- fairness, medical, elderly, and quiet-sensitive features

and it must choose the final solution according to the current offline selection policy.

## Raw Data

The source dataset is:

- `data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/llm_selection_train_1000_qs_v1.jsonl`

Each raw sample mainly contains:

- `selection_input`: the decision problem
- `selection_target`: the label
- `messages`: a chat-style SFT version

Important fields in `selection_input`:

- `problem_context`
- `scene_summary`
- `demand_cards`
- `objective_groups`

Each candidate solution includes features such as:

- distance
- delivery time
- noise impact
- service rate
- weighted priority coverage
- elderly population coverage
- quiet-sensitive service / noise proxy

Important fields in `selection_target`:

- `selected_group_id`
- `selected_solution_id`
- `selection_mode`
- `primary_reason_codes`
- `training_labels`

## Compact Export

Before training, raw samples are compressed by:

- `scripts/export_llm_selection_compact.py`

This step:

- validates and deduplicates records
- splits train / val
- optionally oversamples multi-group samples
- compresses long inputs to reduce prompt length

Current training scripts use the compact config:

- `KEEP_CANDIDATES_PER_GROUP=3`
- `MAX_DEMAND_CARDS=6`

The compact outputs are written under:

- `data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/compact_exports_4b_short`

## What The LLM Sees

The model input is a chat prompt built from the compact selection input.

The user prompt is roughly:

```text
Select one Pareto-frontier solution for the UAV routing problem.
Return JSON with keys: selected_group_id, selected_solution_id, selection_mode,
primary_reason_codes, decision_confidence, training_labels.

Input JSON:
{ ... compact_selection_input ... }
```

So the LLM sees a **structured decision problem**, not raw route geometry.

## SFT Stage

Script:

- `scripts/training_sft_3gpu_llm_selection.sh`

SFT trains the model to imitate the labeled selector output.

Training target:

- output valid JSON
- choose the correct `selected_group_id`
- choose the correct `selected_solution_id`
- optionally match reason codes and scene-related labels

The SFT response target includes:

- `selected_group_id`
- `selected_solution_id`
- `selection_mode`
- `primary_reason_codes`
- `decision_confidence`
- `training_labels`

After SFT, the script automatically merges the LoRA checkpoint into a deployable HuggingFace model with:

- `scripts/loRA_to_merged.py`

## GRPO Stage

Script:

- `scripts/training_grpo_3gpu_llm_selection.sh`

GRPO starts from the merged SFT model and further optimizes the model using rule-based reward.

Reward script:

- `scripts/verl_llm_selection_reward.py`

The reward mainly encourages:

- valid JSON output
- selecting a candidate that actually exists
- matching the correct group
- matching the correct final solution
- partially matching reason codes / scene type

The most important part is the exact match on:

- `selected_solution_id`

After GRPO, the script also merges the latest actor checkpoint into a deployable HuggingFace model.

## Evaluation

Evaluation script:

- `scripts/eval_llm_selection_models.py`

It compares:

- base model
- SFT model
- GRPO model

Main reported metric:

- `solution_accuracy`

This is the exact-match accuracy of:

- predicted `selected_solution_id`
- vs ground-truth `selected_solution_id`

Other useful metrics:

- `group_accuracy`
- `candidate_valid_rate`
- `json_valid_rate`
- `avg_reward`

## One-Line Summary

This pipeline trains an LLM to read a compact, structured Pareto-selection prompt and output the policy-consistent final solution in JSON form, first by supervised imitation (SFT) and then by reward-based refinement (GRPO).
