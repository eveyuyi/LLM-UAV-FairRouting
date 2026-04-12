#!/usr/bin/env python3
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "spearman",
    "kendall_tau",
    "top_k_hit_rate",
    "priority_1_recall",
    "priority_1_f1",
    "urgent_recall",
    "urgent_f1",
]

PRIORITY_METRICS = [
    "priority_1_recall",
    "priority_1_f1",
    "urgent_recall",
    "urgent_f1",
]

CORE_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "spearman",
    "kendall_tau",
]


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def sanitize_num(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{v:.6f}"


def extract_model_from_run_dir(run_dir: str) -> str:
    if not run_dir:
        return ""
    m = re.search(r"run_\d{8}_\d{6}_(.+?)_noise", run_dir)
    if m:
        return m.group(1)
    return Path(run_dir).name


def add_value_labels(ax, bars, fmt="{:.3f}", fontsize=7):
    for b in bars:
        h = b.get_height()
        if h is None:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.01,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def parse_step_from_model_name(model_name: str):
    m = re.search(r"gs(\d+)", model_name or "")
    if not m:
        return None
    return int(m.group(1))


def is_target_experiment_model(model_name: str) -> bool:
    if not model_name:
        return False
    name = model_name.lower()
    return name.startswith("post-expa") or name.startswith("post-expb") or name.startswith("post-expc")


def main():
    repo = Path("/data/code/LLM-UAV-FairRouting")
    eval_root = repo / "data" / "eval_runs"
    out_dir = repo / "results" / "analysis" / "eval_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(eval_root.glob("**/evals/summary.json"))
    rows = []

    for p in summary_paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        rel = p.relative_to(repo)
        run_dir = p.parent.parent.name
        scope = data.get("scope", "")
        mode = data.get("mode", "")
        priority_common = (data.get("priority_modes") or {}).get("common", "")
        primary_overall = (((data.get("verdicts") or {}).get("primary") or {}).get("overall", ""))

        artifacts = data.get("artifacts") or {}
        pre_model = extract_model_from_run_dir(artifacts.get("pre_run_dir", ""))
        post_model = extract_model_from_run_dir(artifacts.get("post_run_dir", ""))

        seed_match = re.search(r"seed_?(\d+)", str(rel))
        seed = seed_match.group(1) if seed_match else ""

        run_key = re.sub(r"seed_?\d+", "seedX", run_dir)
        run_key = re.sub(r"_\d{8}_\d{6}", "", run_key)

        align = data.get("alignment") or {}
        pre = align.get("pre") or {}
        post = align.get("post") or {}
        delta = align.get("delta_post_minus_pre") or {}

        row = {
            "summary_path": str(rel),
            "run_dir": run_dir,
            "run_key": run_key,
            "seed": seed,
            "scope": scope,
            "mode": mode,
            "priority_mode": priority_common,
            "overall": primary_overall,
            "pre_model": pre_model,
            "post_model": post_model,
        }
        for m in METRICS:
            row[f"pre_{m}"] = pre.get(m)
            row[f"post_{m}"] = post.get(m)
            row[f"delta_{m}"] = delta.get(m)
        row["priority_pre_mean"] = mean([pre.get(m) for m in PRIORITY_METRICS])
        row["priority_post_mean"] = mean([post.get(m) for m in PRIORITY_METRICS])
        row["priority_delta_mean"] = mean([delta.get(m) for m in PRIORITY_METRICS])
        row["core_pre_mean"] = mean([pre.get(m) for m in CORE_METRICS])
        row["core_post_mean"] = mean([post.get(m) for m in CORE_METRICS])
        row["core_delta_mean"] = mean([delta.get(m) for m in CORE_METRICS])
        rows.append(row)

    seed_csv = out_dir / "eval_seed_level_table.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_model = defaultdict(list)
    for r in rows:
        key = (r["post_model"], r["priority_mode"], r["scope"])
        by_model[key].append(r)

    model_rows = []
    for (post_model, priority_mode, scope), items in by_model.items():
        agg = {
            "post_model": post_model,
            "priority_mode": priority_mode,
            "scope": scope,
            "n_runs": len(items),
            "n_unique_seeds": len({x["seed"] for x in items if x["seed"]}),
            "pre_model_examples": ";".join(sorted({x["pre_model"] for x in items if x["pre_model"]})[:3]),
            "overall_post_better": sum(1 for x in items if x["overall"] == "post_better"),
            "overall_pre_better": sum(1 for x in items if x["overall"] == "pre_better"),
            "overall_mixed": sum(1 for x in items if x["overall"] == "mixed"),
        }
        for m in METRICS:
            agg[f"pre_{m}_mean"] = mean([x[f"pre_{m}"] for x in items])
            agg[f"post_{m}_mean"] = mean([x[f"post_{m}"] for x in items])
            agg[f"delta_{m}_mean"] = mean([x[f"delta_{m}"] for x in items])
        agg["priority_pre_mean"] = mean([x["priority_pre_mean"] for x in items])
        agg["priority_post_mean"] = mean([x["priority_post_mean"] for x in items])
        agg["priority_delta_mean"] = mean([x["priority_delta_mean"] for x in items])
        agg["core_pre_mean"] = mean([x["core_pre_mean"] for x in items])
        agg["core_post_mean"] = mean([x["core_post_mean"] for x in items])
        agg["core_delta_mean"] = mean([x["core_delta_mean"] for x in items])
        model_rows.append(agg)

    model_rows.sort(key=lambda x: (x["priority_mode"], x["scope"], -(x["priority_post_mean"] or -999)))

    model_csv = out_dir / "eval_model_means_table.csv"
    model_fields = list(model_rows[0].keys()) if model_rows else []
    with model_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=model_fields)
        writer.writeheader()
        writer.writerows(model_rows)

    hard_rows = [
        r
        for r in model_rows
        if r["priority_mode"] == "llm-only"
        and is_target_experiment_model(r["post_model"])
        and "hard_eval_v1" in "".join([x["summary_path"] for x in rows if x["post_model"] == r["post_model"]])
    ]
    unique = {}
    for r in hard_rows:
        unique[r["post_model"]] = r
    hard_rows = list(unique.values())
    hard_rows.sort(key=lambda x: x["priority_post_mean"] or -999, reverse=True)

    if hard_rows:
        top = hard_rows[:20]
        names = [x["post_model"] for x in top]
        vals = [x["priority_post_mean"] or 0.0 for x in top]
        fig, ax = plt.subplots(figsize=(16, 8))
        bars = ax.bar(range(len(vals)), vals, color="#1f77b4")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=75, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean Absolute Score (Post)")
        ax.set_title("Model Ranking by Priority Post Mean (hard eval, llm-only)")
        add_value_labels(ax, bars, fmt="{:.3f}", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "priority_post_ranking_hard_llmonly.png", dpi=180)
        plt.close()

        top2 = hard_rows[:12]
        names2 = [x["post_model"] for x in top2]
        x = list(range(len(names2)))
        width = 0.2
        fig, ax = plt.subplots(figsize=(18, 8))
        for i, m in enumerate(PRIORITY_METRICS):
            vals_m = [r[f"post_{m}_mean"] or 0.0 for r in top2]
            bars = ax.bar([xi + (i - 1.5) * width for xi in x], vals_m, width=width, label=m)
            add_value_labels(ax, bars, fmt="{:.3f}", fontsize=6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names2, rotation=75, ha="right")
        ax.set_ylabel("Absolute Score (Post)")
        ax.set_title("Priority Metric Post Scores by Model (Top 12, hard eval llm-only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "priority_metrics_post_top_models_hard_llmonly.png", dpi=180)
        plt.close()

        # All absolute post metrics (10 metrics) for top models.
        step_sorted_rows = []
        for r in hard_rows:
            step = parse_step_from_model_name(r["post_model"])
            if step is None:
                continue
            step_sorted_rows.append((step, r))
        step_sorted_rows.sort(key=lambda x: x[0])

        top3 = [x[1] for x in step_sorted_rows[:12]]
        names3 = [x["post_model"] for x in top3]
        fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
        axes = axes.flatten()
        for i, m in enumerate(METRICS):
            axm = axes[i]
            vals_m = [r.get(f"post_{m}_mean") or 0.0 for r in top3]
            bars = axm.bar(range(len(names3)), vals_m, color="#4e79a7")
            add_value_labels(axm, bars, fmt="{:.3f}", fontsize=6)
            axm.set_title(m)
            axm.set_ylim(0, 1.0)
            axm.set_xticks(range(len(names3)))
            axm.set_xticklabels(names3, rotation=75, ha="right", fontsize=7)
        fig.suptitle("All Post Metrics by Checkpoint Step (hard eval, llm-only)")
        fig.tight_layout()
        fig.savefig(out_dir / "all_metrics_post_top_models_hard_llmonly.png", dpi=180)
        plt.close()

        # Relative trend view: each metric normalized by first checkpoint in step order.
        if top3:
            steps3 = [parse_step_from_model_name(r["post_model"]) or 0 for r in top3]
            fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=True)
            axes = axes.flatten()
            for i, m in enumerate(METRICS):
                axm = axes[i]
                vals_m = [r.get(f"post_{m}_mean") or 0.0 for r in top3]
                base = vals_m[0] if vals_m else 0.0
                if base == 0:
                    rel = [0.0 for _ in vals_m]
                else:
                    rel = [v / base for v in vals_m]
                axm.plot(steps3, rel, marker="o", linewidth=2.0, color="#e15759")
                axm.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
                for sx, sy in zip(steps3, rel):
                    axm.text(sx, sy + 0.01, f"{sy:.3f}", ha="center", va="bottom", fontsize=6)
                axm.set_title(m)
                axm.grid(True, alpha=0.3)
                axm.set_xlabel("Training Step (gs)")
                axm.set_ylabel("Relative to first step")
            fig.suptitle("All Metrics Relative Trend vs Step (hard eval, llm-only)")
            fig.tight_layout()
            fig.savefig(out_dir / "all_metrics_relative_trend_hard_llmonly.png", dpi=180)
            plt.close()

        # Clear checkpoint selection view:
        # x=core post mean, y=priority post mean, labels=checkpoint names.
        fig, ax = plt.subplots(figsize=(11, 8))
        xs = [r["core_post_mean"] or 0.0 for r in hard_rows]
        ys = [r["priority_post_mean"] or 0.0 for r in hard_rows]
        ns = [r["n_runs"] or 1 for r in hard_rows]
        # bubble size lightly reflects available runs
        sizes = [40 + 25 * n for n in ns]
        ax.scatter(xs, ys, s=sizes, alpha=0.8, color="#2a9d8f", edgecolor="black", linewidth=0.5)
        for r, x0, y0 in zip(hard_rows, xs, ys):
            ax.text(x0 + 0.0015, y0 + 0.0015, r["post_model"], fontsize=7)
        ax.set_xlabel("Core Post Mean")
        ax.set_ylabel("Priority Post Mean")
        ax.set_title("Checkpoint Selection Map (hard eval, llm-only)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(xs) - 0.01, max(xs) + 0.01)
        ax.set_ylim(min(ys) - 0.01, max(ys) + 0.01)
        fig.tight_layout()
        fig.savefig(out_dir / "ckpt_selection_map_hard_llmonly.png", dpi=180)
        plt.close()

        # Experiment C (and C8) step-vs-metrics multi-subplots.
        expc_rows = [r for r in hard_rows if (r["post_model"] or "").lower().startswith("post-expc")]
        expc_rows_with_step = []
        for r in expc_rows:
            step = parse_step_from_model_name(r["post_model"])
            if step is not None:
                expc_rows_with_step.append((step, r))
        expc_rows_with_step.sort(key=lambda x: x[0])

        if expc_rows_with_step:
            steps = [x[0] for x in expc_rows_with_step]
            fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=True)
            axes = axes.flatten()
            for i, m in enumerate(METRICS):
                axm = axes[i]
                ys_m = [x[1].get(f"post_{m}_mean") or 0.0 for x in expc_rows_with_step]
                axm.plot(steps, ys_m, marker="o", linewidth=2.0, color="#4e79a7")
                for sx, sy in zip(steps, ys_m):
                    axm.text(sx, sy + 0.01, f"{sy:.3f}", ha="center", va="bottom", fontsize=6)
                axm.set_title(m)
                axm.set_ylim(0, 1.0)
                axm.grid(True, alpha=0.3)
            for axm in axes:
                axm.set_xlabel("Training Step (gs)")
            fig.suptitle("Experiment C Metrics vs Training Step (hard eval, llm-only)")
            fig.tight_layout()
            fig.savefig(out_dir / "experimentC_metrics_vs_step_hard_llmonly.png", dpi=180)
            plt.close()

    md = []
    md.append("# Evaluation Summary Table and Plots")
    md.append("")
    md.append(f"- Total summary files parsed: **{len(rows)}**")
    md.append(f"- Seed-level table: `{seed_csv.relative_to(repo)}`")
    md.append(f"- Model-mean table: `{model_csv.relative_to(repo)}`")
    md.append(f"- Plot 1: `{(out_dir / 'priority_post_ranking_hard_llmonly.png').relative_to(repo)}`")
    md.append(f"- Plot 2: `{(out_dir / 'priority_metrics_post_top_models_hard_llmonly.png').relative_to(repo)}`")
    md.append(f"- Plot 3: `{(out_dir / 'all_metrics_post_top_models_hard_llmonly.png').relative_to(repo)}`")
    md.append(f"- Plot 4: `{(out_dir / 'ckpt_selection_map_hard_llmonly.png').relative_to(repo)}`")
    md.append(f"- Plot 5: `{(out_dir / 'experimentC_metrics_vs_step_hard_llmonly.png').relative_to(repo)}`")
    md.append(f"- Plot 6: `{(out_dir / 'all_metrics_relative_trend_hard_llmonly.png').relative_to(repo)}`")
    md.append("")
    md.append("## Top Models by Priority Post Mean (hard eval, llm-only)")
    md.append("")
    md.append("| Rank | post_model | priority_post_mean | core_post_mean | priority_pre_mean | post_better/pre_better/mixed |")
    md.append("|---|---|---:|---:|---:|---|")
    for i, r in enumerate(hard_rows[:20], start=1):
        md.append(
            f"| {i} | `{r['post_model']}` | {sanitize_num(r['priority_post_mean'])} | {sanitize_num(r['core_post_mean'])} | "
            f"{sanitize_num(r['priority_pre_mean'])} | "
            f"{r['overall_post_better']}/{r['overall_pre_better']}/{r['overall_mixed']} |"
        )
    (out_dir / "eval_summary_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
