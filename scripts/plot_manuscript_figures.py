#!/usr/bin/env python
"""Generate only the figures currently used by docs/experiments_section.tex.

This is a short manuscript-facing entry point. The longer
plot_experiment_figures.py file keeps the historical exploratory plots and
helper implementations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_experiment_figures import (  # noqa: E402
    OUT_DIR,
    plot_experiment_testbed,
    plot_fleet_sensitivity,
    plot_main_pareto_combo_quad,
    plot_priority_alignment,
    setup_style,
)


FIGURES = {
    "testbed": plot_experiment_testbed,
    "main": plot_main_pareto_combo_quad,
    "fleet": plot_fleet_sensitivity,
    "priority": plot_priority_alignment,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures",
        nargs="*",
        default=None,
        help="Figure(s) to generate. Default: all manuscript figures.",
    )
    args = parser.parse_args()

    requested = args.figures or ["all"]
    invalid = [name for name in requested if name not in FIGURES and name != "all"]
    if invalid:
        choices = ", ".join(sorted(FIGURES) + ["all"])
        parser.error(f"invalid figure(s): {', '.join(invalid)}. Choose from: {choices}")
    selected = list(FIGURES) if "all" in requested else requested
    setup_style()
    for name in selected:
        FIGURES[name]()
    print(f"Wrote {', '.join(selected)} figure(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
