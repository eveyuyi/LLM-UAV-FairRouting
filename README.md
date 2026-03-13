# llm4fairrouting

`llm4fairrouting` is a drone-delivery research project that combines an LLM-driven demand understanding workflow with a shared dynamic routing and CPLEX-based solving core. The repository supports both:

- an end-to-end workflow from demand events to routing results
- a baseline demo that directly uses the CPLEX-based solving core

## Overview

The end-to-end LLM4fairrouting workflow is organized into three modules:

1. `Module 1`: convert structured demand events into dialogue-style requests
2. `Module 2`: extract structured delivery demands from dialogues
3. `Module 3`: infer priority/weight settings and solve dynamic routing windows

`Module 3` is implemented as two connected stages:

- `Module 3a`: priority / weight inference
- `Module 3b`: solver adapter and routing execution

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

Run the workflow with:

```bash
./scripts/run_workflow.sh
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
| Module 1 | `src/llm4fairrouting/llm/dialogue_generation.py` | Generate dialogue-style requests from structured demand events | `data/seed/daily_demand_events.csv`, optional station file | `generated_dialogues.jsonl` |
| Module 2 | `src/llm4fairrouting/llm/demand_extraction.py` | Extract structured delivery demands from dialogues by time window | `generated_dialogues.jsonl` | `extracted_demands.json` |
| Module 3 | `src/llm4fairrouting/llm/priority_inference.py`, `src/llm4fairrouting/workflow/solver_adapter.py` | Infer per-demand priority settings and solve routing windows | `extracted_demands.json`, weight configs, station/building data | `weight_configs.json` / `weight_configs/`, `solver_results.json`, `workflow_results.json` |

#### Module 1

- File: `src/llm4fairrouting/llm/dialogue_generation.py`
- Role: converts structured demand-event records into dialogue data for downstream modules
- Input:
  - `daily_demand_events.csv`
  - optional station metadata from `drone_station_locations.csv`
- Output:
  - standalone default: `data/drone/generated_dialogues.jsonl`
  - workflow run: `results/run_*/generated_dialogues.jsonl`

#### Module 2

- File: `src/llm4fairrouting/llm/demand_extraction.py`
- Role: groups dialogues by time window and extracts solver-ready structured demands
- Input:
  - `generated_dialogues.jsonl`
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

### Baseline

The baseline entry is:

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

### Shared Routing Core

The routing core is shared by the workflow and the baseline:

- `src/llm4fairrouting/routing/domain.py`: core entities such as `Point`, `DemandEvent`, and drone states
- `src/llm4fairrouting/routing/path_costs.py`: path planning, distance computation, and noise-cost estimation
- `src/llm4fairrouting/routing/assignment_model.py`: Pyomo/CPLEX assignment model
- `src/llm4fairrouting/routing/simulator.py`: dynamic simulator for time-evolving demand and drone execution
- `src/llm4fairrouting/routing/serialization.py`: serialization helpers for simulation and workflow results


