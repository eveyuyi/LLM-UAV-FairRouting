# llm4fairrouting

This project unifies the LLM workflow and the CPLEX baseline around one shared dynamic routing core.

## Structure

- `src/llm4fairrouting/llm/`: dialogue generation, demand extraction, priority inference
- `src/llm4fairrouting/workflow/`: workflow runner and solver adapter
- `src/llm4fairrouting/routing/`: shared dynamic routing core used by workflow and baseline
- `src/llm4fairrouting/data/`: building/station loaders and seed paths
- `src/llm4fairrouting/baselines/`: baseline/demo scripts built on the shared routing core
- `tests/`: unit tests
- `data/seed/`: seed datasets

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

Run the integrated workflow:

```bash
python -m llm4fairrouting.workflow.run_workflow --offline --skip-solver
```

You can also keep your API key and common experiment parameters in a local `.env`
file. Copy [`.env.example`](/Users/eveyu/Downloads/githubs/drone-delivery-pipeline/.env.example) to `.env`,
fill in the values once, then run:

```bash
./scripts/run_workflow.sh
```

The workflow CLI auto-loads `.env` from the project root, so even without the
helper script you can shorten manual runs to:

```bash
PYTHONPATH=src conda run -n gpt_academic python -m llm4fairrouting.workflow.run_workflow
```

Run each module independently:

```bash
python -m llm4fairrouting.llm.dialogue_generation --offline
python -m llm4fairrouting.llm.demand_extraction --offline
python -m llm4fairrouting.llm.priority_inference --offline
python -m llm4fairrouting.workflow.solver_adapter --weights path/to/weight_configs.json
python -m llm4fairrouting.baselines.cplex_with_priority_noise
```

If the project is installed with `pip install -e .`, the same entrypoints are also
available as console commands:

```bash
llm4fairrouting-dialogue
llm4fairrouting-demand
llm4fairrouting-priority
llm4fairrouting-solver
llm4fairrouting-workflow
```
