# Drone Delivery Pipeline

This is an independent project extracted from LLMOPT for drone medical delivery workflow experiments.

## Structure

- `src/drone_pipeline/pipeline/`: module 1/2/3 pipeline code
- `src/drone_pipeline/scripts/`: per-module script entrypoints and integrated experiment runner
- `src/drone_pipeline/prompts/`: prompt templates
- `src/drone_pipeline/utils/`: solver and data utilities
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

Run the integrated experiment pipeline:

```bash
python -m drone_pipeline.scripts.run_pipeline --offline --skip-solver
```

Run each module independently:

```bash
python -m drone_pipeline.scripts.module1_generate_dialogues --offline
python -m drone_pipeline.scripts.module2_extract_demands --offline
python -m drone_pipeline.scripts.module3_adjust_weights --offline
python -m drone_pipeline.scripts.module3_solve --weights path/to/weight_configs.json
```

If the project is installed with `pip install -e .`, the same entrypoints are also
available as console commands:

```bash
drone-module1
drone-module2
drone-module3a
drone-module3b
drone-pipeline
```
