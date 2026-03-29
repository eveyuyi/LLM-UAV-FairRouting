# llm4fairrouting

`llm4fairrouting` is a drone-delivery research project that combines an LLM-driven demand understanding workflow with a shared dynamic routing and CPLEX-based solving core. The repository supports both:

- an end-to-end workflow from demand events to routing results
- a baseline demo that directly uses the CPLEX-based solving core

## Overview

The end-to-end LLM4fairrouting workflow is organized into three modules:

1. `Module 1`: build a fixed LLM-generated dialogue dataset from structured demand events
2. `Module 2`: extract structured delivery demands from dialogues
3. `Module 3`: infer priority/weight settings and solve dynamic routing windows

The data-generation stack also supports a richer training path for LLM2/LLM3 with:

- seed event CSVs for solver compatibility
- rich event manifests carrying `latent_priority`, observability factors, and gold extraction labels
- multi-style dialogues plus automatic dialogue audit
- extracted demands labeled with `extraction_observable_priority`
- training corpora split into `clean_structured`, `pipeline_structured`, and `hard_contrastive`

`Module 3` is implemented as two connected stages:

- `Module 3a`: priority / weight inference
- `Module 3b`: solver adapter and routing execution

•	提取是否准：precision，recall，f1，exact_match_rate
•	优先级是否准：accuracy，macro_f1，weighted_f1，confusion_matrix
•	规划结果是否好：service_rate，service_rate_loss，final_total_distance_m，final_total_noise_impact，average_delivery_time_h，max_delivery_time_h，n_used_drones
•	帕累托前沿是否有代表性、搜索是否高效：final_total_distance_m，average_delivery_time_h，final_total_noise_impact，service_rate_loss，n_used_drones
•	搜索过程指标frontier_size，n_candidates_evaluated，search_runtime_s，avg_candidate_runtime_s

## Project Workflow Diagram

![Project Workflow](docs/images/workflow.png)



## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Copy the example environment file first:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running any online Module 1/2/3 command. If you use a compatible gateway, set `OPENAI_BASE_URL` too.

Build the canonical dialogue dataset with:

```bash
./scripts/build_daily_demand_dialogues.sh
```

Run the llm4fairrouting workflow with:

```bash
./scripts/run_workflow.sh
```

Run the baseline with:

```
PYTHONPATH=src python -m llm4fairrouting.baselines.cplex_with_priority_noise
```

Generate the primary rich event manifest with:

```bash
llm4fairrouting-demand-events
```

Generate the richer LLM2/LLM3 training dataset with:

```bash
llm4fairrouting-training-data
```

If the package is not installed in editable mode yet, use:

```bash
PYTHONPATH=src python -m llm4fairrouting.data.demand_event_generation
```


## Repository Layout

- `src/llm4fairrouting/llm/`: LLM-related modules for dialogue generation, demand extraction, and priority inference
- `src/llm4fairrouting/workflow/`: end-to-end workflow entry and solver adapter
- `src/llm4fairrouting/routing/`: shared routing core used by both workflow and baseline
- `src/llm4fairrouting/data/`: seed-path definitions and dataset loaders
- `src/llm4fairrouting/baselines/`: baseline entry points
- `src/llm4fairrouting/config/`: runtime environment and `.env` helpers
- `data/seed/`: seed datasets
- `results/`: workflow run outputs
- `scripts/`: shell wrappers for running the workflow
- `tests/`: unit tests
- `docs/images/`: recommended location for README and documentation figures

### Workflow Modules

| Module | Location | Purpose | Main Input | Main Output |
| --- | --- | --- | --- | --- |
| Module 1 | `src/llm4fairrouting/data/demand_dialogue_dataset.py`, `src/llm4fairrouting/llm/dialogue_generation.py` | Build the fixed seed dialogue dataset from structured demand events with an LLM | `data/seed/daily_demand_events_manifest.jsonl`, optional station file | `data/seed/daily_demand_dialogues.jsonl` |
| Module 2 | `src/llm4fairrouting/llm/demand_extraction.py` | Extract structured delivery demands from dialogues by time window | `data/seed/daily_demand_dialogues.jsonl` | `extracted_demands.json` |
| Module 3 | `src/llm4fairrouting/llm/priority_inference.py`, `src/llm4fairrouting/workflow/solver_adapter.py` | Infer per-demand priority settings and solve routing windows | `extracted_demands.json`, weight configs, station/building data | `weight_configs.json` / `weight_configs/`, `solver_results.json`, `workflow_results.json` |
| Training Builder | `src/llm4fairrouting/data/training_dataset_builder.py` | Build LLM2/LLM3 training data with observable priority labels and hard contrastive windows | rich event manifest or generated seed events | `data/seed/priority_training_dataset.json` |

### Demand Event Generation

- File: `src/llm4fairrouting/data/demand_event_generation.py`
- Role: generates seed demand events from `building_information.csv`, emits a rich manifest for the latest LLM2/LLM3 data flow, and can optionally project a legacy CSV
- Defaults:
  - 5-minute windows across a full day
  - 4-10 demands per window
  - `medical_ratio=0.2`, so about 20% `medical` and 80% `commercial`
  - `medical` demands use priorities `1/2/3` with equal probability; `commercial` demands use priority `4`
  - manifest output carries `latent_priority`, observability factors, gold structured demand, and solver-useful labels
  - optional CSV projection still uses normalized English values such as `medical` and `commercial`
- Run:

```bash
llm4fairrouting-demand-events --manifest-output data/seed/daily_demand_events_manifest.jsonl
```

Optional rich manifest output:

```bash
llm4fairrouting-demand-events --manifest-output data/seed/daily_demand_events_manifest.jsonl
```

#### Module 1

- Files: `src/llm4fairrouting/data/demand_dialogue_dataset.py`, `src/llm4fairrouting/llm/dialogue_generation.py`
- Role: builds one fixed LLM-generated dialogue dataset aligned with the seed demand events
- Input:
  - `daily_demand_events_manifest.jsonl`
  - optional station metadata from `drone_station_locations.csv`
- Output:
  - canonical seed dataset: `data/seed/daily_demand_dialogues.jsonl`

#### Module 2

- File: `src/llm4fairrouting/llm/demand_extraction.py`
- Role: groups dialogues by time window and extracts solver-ready structured demands
- Input:
  - `daily_demand_dialogues.jsonl`
- Output:
  - standalone default: `data/drone/extracted_demands.json`
  - workflow run: `results/run_*/extracted_demands.json`

#### Module 3

##### Module 3a: Priority / Weight Inference

- File: `src/llm4fairrouting/llm/priority_inference.py`
- Role: assigns priority, window rank, and supplementary constraints for each demand
- Input:
  - `extracted_demands.json`
- Output:
  - standalone default: `data/drone/weight_configs.json`
  - workflow run: `results/run_*/weight_configs/weight_config_window*.json`

##### Module 3b: Solver Adapter

- File: `src/llm4fairrouting/workflow/solver_adapter.py`
- Role: transforms Module 2 + Module 3a outputs into shared routing-core inputs and executes dynamic solving
- Input:
  - `extracted_demands.json`
  - `weight_configs.json` or `weight_configs/`
  - `data/seed/drone_station_locations.csv`
  - `data/seed/building_information.csv`
- Output:
  - standalone default: `data/drone/solver_results.json`
  - workflow run: `results/run_*/workflow_results.json`
  - analytics sidecar: `*_analytics/solver_analytics.json`, `*_analytics/charts/*.png`

### Baseline

The original baseline entry is:

- file: `src/llm4fairrouting/baselines/cplex_with_priority_noise.py`
- function: `main()`

Run the baseline with:

```bash
PYTHONPATH=src python -m llm4fairrouting.baselines.cplex_with_priority_noise
```

The baseline:

- does not use the LLM pipeline
- loads seed building and station data directly
- generates demand events internally
- reuses the shared routing core in `src/llm4fairrouting/routing/`

An additional seed-aligned baseline is also available:

- file: `src/llm4fairrouting/baselines/cplex_with_seed_priorities.py`
- role: reads `data/seed/daily_demand_events_manifest.jsonl` directly and solves with manifest-aligned priority labels

### Shared Routing Core

The routing core is shared by the workflow and the baseline:

- `src/llm4fairrouting/routing/domain.py`: core entities such as `Point`, `DemandEvent`, and drone states
- `src/llm4fairrouting/routing/path_costs.py`: path planning, distance computation, and noise-cost estimation
- `src/llm4fairrouting/routing/assignment_model.py`: Pyomo/CPLEX assignment model
- `src/llm4fairrouting/routing/simulator.py`: dynamic simulator for time-evolving demand and drone execution
- `src/llm4fairrouting/routing/serialization.py`: serialization helpers for simulation and workflow results
- `src/llm4fairrouting/routing/analytics.py`: convergence traces, Pareto scans, and chart export

### Solver Analytics Outputs

When the solver runs, the project can now export a sidecar analytics directory with:

- `solver_analytics.json`: per-solve model size, objective breakdown, incumbent trace, Gantt tasks, and run summary
- `charts/convergence_curve.png`: incumbent objective improvements over solve time
- `charts/solve_time_vs_problem_size.png`: relationship between solve time and model size
- `charts/drone_schedule_gantt.png`: executed drone schedule
- `pareto/pareto_frontier.json` and `pareto/charts/pareto_frontier.png`: weighted-sum multi-objective scan outputs when `--pareto-scan` is enabled

Useful flags:

```bash
python -m llm4fairrouting.workflow.run_workflow --offline --pareto-scan --enable-conflict-refiner
```
