# Drone Delivery Pipeline

This is an independent project extracted from LLMOPT for drone medical delivery workflow experiments.

## Structure

- `src/drone_pipeline/pipeline/`: module 1/2/3 pipeline code
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

Run offline pipeline:

```bash
python -m drone_pipeline.pipeline.run_pipeline --offline --skip-solver
```
