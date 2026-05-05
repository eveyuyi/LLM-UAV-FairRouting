#!/usr/bin/env python
"""Generate publication-style experiment figures for the LLM4FairRouting paper.

The script reads the checked-in evaluation JSON files and writes PDF, PNG, and
SVG figures under docs/images/experiments. It intentionally uses matplotlib's
ggplot style plus a ggplot2-like discrete palette so the figures remain easy to
regenerate in the base environment.
"""

from __future__ import annotations

import io
import json
import math
import os
import textwrap
from pathlib import Path

MPL_CONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/llm4fairrouting-mplconfig"))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images" / "experiments_revised"

METHOD_ORDER = ["M0a", "M0b", "M0c", "M1_pre", "M1_gemini", "M1_ft"]
METHOD_LABELS = {
    "M0a": "Uninformative priority control",
    "M0b": "No-priority control",
    "M0c": "Rule-based control",
    "M1_gemini": "gemini-3.1-flash-lite",
    "M1_pre": "Qwen3-4b",
    "M1_sft": "Qwen3-4b-aligned-SFT (ours)",
    "M1_ft": "Qwen3-4b-aligned (ours)",
}

# ggplot-inspired palette: controls are neutral/coral, Gemini is magenta, and
# Qwen variants stay within a broad cyan-blue-indigo family.
GG_COLORS = {
    "M0a": "#999999",
    "M0b": "#BDBDBD",
    "M0c": "#F8766D",
    "M1_gemini": "#E76BF3",
    "M1_pre": "#00BFC4",
    "M1_sft": "#619CFF",
    "M1_ft": "#3B4CC0",
}

PRIORITY_COLORS = {
    1: "#F8766D",
    2: "#D89000",
    3: "#00BFC4",
    4: "#7CAE00",
}

SERVICE_TIER_LABELS = {
    1: "1st-priority",
    2: "2nd-priority",
    3: "3rd-priority",
    4: "4th-priority",
}

PRIORITY_WEIGHTS = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}


def setup_style() -> None:
    plt.style.use("ggplot")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "#F7F7F7",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: str | Path) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(ROOT / path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_figure(fig: mpl.figure.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axes(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="white", linewidth=1.1)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    for spine in ax.spines.values():
        spine.set_visible(False)


def route_methods(path: str) -> dict:
    return load_json(path)["methods"]


def service_on_time_rate(method_data: dict, tiers: list[int]) -> float:
    """Weighted on-time rate for a set of service tiers."""
    total = 0
    on_time = 0.0
    for tier in tiers:
        row = method_data["by_priority"][str(tier)]
        total += row["count"]
        on_time += row["count"] * row["on_time_rate"]
    return on_time / total if total else 0.0


def method_label(method: str, width: int | None = None) -> str:
    label = METHOD_LABELS[method]
    return textwrap.fill(label, width=width) if width else label


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lon(x: int, zoom: int) -> float:
    return x / (2 ** zoom) * 360.0 - 180.0


def _tile_to_lat(y: int, zoom: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (2 ** zoom)
    return math.degrees(math.atan(math.sinh(n)))


def _fetch_osm_tile(zoom: int, x: int, y: int) -> Image.Image | None:
    cache_path = OUT_DIR / "_osm_tiles" / str(zoom) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "LLM4FairRouting-paper-figure/0.1"},
        )
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def _osm_basemap(
    bbox: tuple[float, float, float, float],
    zoom: int = 11,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    lon_min, lon_max, lat_min, lat_max = bbox
    x_min, y_min = _lonlat_to_tile(lon_min, lat_max, zoom)
    x_max, y_max = _lonlat_to_tile(lon_max, lat_min, zoom)

    canvas = Image.new("RGB", ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256), "#F4F4F4")
    fetched = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile = _fetch_osm_tile(zoom, x, y)
            if tile is None:
                continue
            canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
            fetched += 1

    if fetched == 0:
        return None

    extent = (
        _tile_to_lon(x_min, zoom),
        _tile_to_lon(x_max + 1, zoom),
        _tile_to_lat(y_max + 1, zoom),
        _tile_to_lat(y_min, zoom),
    )
    return np.asarray(canvas), extent


def plot_main_results() -> None:
    """Main result figure for the primary seven-UAV continuous simulation."""
    data = route_methods("evals/results/routeA_7uav_seed4111.json")
    plot_methods = ["M0a", "M0b", "M1_pre", "M1_gemini", "M1_ft"]
    metrics = [
        (
            "First-priority demand",
            [data[m]["by_priority"]["1"]["on_time_rate"] * 100 for m in plot_methods],
            "%",
            (35, 102),
            data["M0c"]["by_priority"]["1"]["on_time_rate"] * 100,
        ),
        (
            "Second-priority demand",
            [data[m]["by_priority"]["2"]["on_time_rate"] * 100 for m in plot_methods],
            "%",
            (45, 102),
            data["M0c"]["by_priority"]["2"]["on_time_rate"] * 100,
        ),
    ]
    time_metrics = [
        (
            "First-priority demand",
            [data[m]["by_priority"]["1"]["avg_latency_min"] for m in plot_methods],
            " min",
            (0, 78),
            data["M0c"]["by_priority"]["1"]["avg_latency_min"],
        ),
        (
            "Second-priority demand",
            [data[m]["by_priority"]["2"]["avg_latency_min"] for m in plot_methods],
            " min",
            (0, 78),
            data["M0c"]["by_priority"]["2"]["avg_latency_min"],
        ),
    ]
    labels = [method_label(m, width=24) for m in plot_methods]
    y = np.arange(len(plot_methods))
    height = 0.31
    offsets = [-height / 1.7, height / 1.7]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)

    for ax, metric_group, xlabel, title in [
        (axes[0], metrics, "On-time rate (%)", "Demand on-time rate"),
        (axes[1], time_metrics, "Delivery time (min)", "Demand delivery time"),
    ]:
        for offset, (priority_label, values, unit, xlim, rule_value) in zip(offsets, metric_group):
            is_first_priority = priority_label.startswith("First")
            rule_color = "#4A4A4A" if is_first_priority else "#8A8A8A"
            rule_dash = (0, (5, 2)) if is_first_priority else (0, (2, 2))
            bars = ax.barh(
                y + offset,
                values,
                height=height,
                color=[GG_COLORS[m] for m in plot_methods],
                edgecolor="white",
                linewidth=0.9,
                hatch=None if is_first_priority else "////",
                alpha=0.96,
            )
            ax.axvline(rule_value, color=rule_color, linestyle=rule_dash, linewidth=1.0, alpha=0.78)
            for bar, value in zip(bars, values):
                near_right = value > xlim[1] - 10
                xpos = value + 1.0
                ha = "left"
                if near_right:
                    xpos = min(value - 1.2, xlim[1] - 1.2)
                    ha = "right"
                ax.text(
                    xpos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}{unit}",
                    va="center",
                    ha=ha,
                    fontsize=7.4,
                    color="#222222",
                    zorder=6,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                )
        ax.set_xlim(*metric_group[0][3])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        style_axes(ax, grid_axis="x")

    axes[0].set_ylabel("Dispatch policy")
    axes[1].tick_params(axis="y", labelleft=False)
    priority_handles = [
        mpl.patches.Patch(facecolor="#D6D6D6", edgecolor="#666666", label="First-priority demand"),
        mpl.patches.Patch(facecolor="#D6D6D6", edgecolor="#666666", hatch="////", label="Second-priority demand"),
        mpl.lines.Line2D([0], [0], color="#4A4A4A", linestyle=(0, (5, 2)), linewidth=1.0, label="Rule-based, first-priority"),
        mpl.lines.Line2D([0], [0], color="#8A8A8A", linestyle=(0, (2, 2)), linewidth=1.0, label="Rule-based, second-priority"),
    ]
    axes[0].legend(handles=priority_handles, frameon=True, loc="lower right", fontsize=7.1)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.82, bottom=0.18, wspace=0.20)
    fig.suptitle("Value-aligned priority improves high-priority delivery service", x=0.02, ha="left")
    save_figure(fig, "fig_main_results")


def plot_experiment_testbed() -> None:
    events = [
        row
        for row in load_jsonl("data/test/test_seeds/norm_eval/seed_4111/events_manifest.jsonl")
        if int(row.get("time_slot", 9999)) < 48
    ]
    demand_rows = []
    origin_rows = []
    for event in events:
        dest_lon, dest_lat = event["destination"]["coords"]
        orig_lon, orig_lat = event["origin"]["coords"]
        demand_rows.append(
            {
                "lon": dest_lon,
                "lat": dest_lat,
                "priority": int(event["latent_priority"]),
                "cargo_type": event["cargo"]["type"],
            }
        )
        origin_rows.append(
            {
                "lon": orig_lon,
                "lat": orig_lat,
                "fid": event["origin"]["fid"],
                "supply_type": event["origin"].get("supply_type", "supply"),
            }
        )

    demands = pd.DataFrame(demand_rows)
    origins = pd.DataFrame(origin_rows).drop_duplicates("fid")
    buildings = pd.read_csv(
        ROOT / "data/seed/building_information.csv",
        usecols=["longitude", "latitude", "land_use_type"],
    )
    local_buildings = buildings.sample(min(len(buildings), 7000), random_state=4112)
    lon_min, lon_max = buildings["longitude"].min(), buildings["longitude"].max()
    lat_min, lat_max = buildings["latitude"].min(), buildings["latitude"].max()
    pad_lon = 0.008
    pad_lat = 0.004

    fig, (ax_city, ax) = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.18},
    )

    shenzhen_bbox = (113.62, 114.20, 22.42, 22.86)
    basemap = _osm_basemap(shenzhen_bbox, zoom=11)
    ax_city.set_xlim(shenzhen_bbox[0], shenzhen_bbox[1])
    ax_city.set_ylim(shenzhen_bbox[2], shenzhen_bbox[3])
    if basemap is not None:
        image, extent = basemap
        ax_city.imshow(image, extent=extent, origin="upper", zorder=0)
    else:
        ax_city.add_patch(
            mpl.patches.Rectangle(
                (shenzhen_bbox[0], shenzhen_bbox[2]),
                shenzhen_bbox[1] - shenzhen_bbox[0],
                shenzhen_bbox[3] - shenzhen_bbox[2],
                facecolor="#F4F4F4",
                edgecolor="#999999",
                linewidth=0.9,
                zorder=0,
            )
        )
    ax_city.set_aspect(1 / math.cos(math.radians((shenzhen_bbox[2] + shenzhen_bbox[3]) / 2)), adjustable="box")
    ax_city.add_patch(
        mpl.patches.Rectangle(
            (lon_min - pad_lon, lat_min - pad_lat),
            (lon_max - lon_min) + 2 * pad_lon,
            (lat_max - lat_min) + 2 * pad_lat,
            fill=False,
            edgecolor="#F8766D",
            linewidth=1.5,
        )
    )
    ax_city.text(0.97, 0.04, "Map data: OpenStreetMap", transform=ax_city.transAxes, fontsize=5.8, color="#555555", ha="right")
    ax_city.set_xlabel("Longitude")
    ax_city.set_ylabel("Latitude")
    ax_city.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax_city.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    style_axes(ax_city, grid_axis="both")

    ax.scatter(
        local_buildings["longitude"],
        local_buildings["latitude"],
        s=4,
        c="#CFCFCF",
        alpha=0.28,
        linewidths=0,
        label="Building point",
    )

    for pri, sub in demands.groupby("priority"):
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=28,
            c=PRIORITY_COLORS[pri],
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            label=SERVICE_TIER_LABELS[pri],
        )
    ax.scatter(
        origins["lon"],
        origins["lat"],
        s=90,
        c="#222222",
        marker="*",
        edgecolors="white",
        linewidths=0.5,
        label="Supply depot",
        zorder=5,
    )
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.legend(ncol=3, loc="lower left", frameon=True, fontsize=7.2)
    style_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.17, wspace=0.18)
    fig.text(0.285, 0.035, "(a)", fontsize=11.2, fontweight="bold", ha="center")
    fig.text(0.755, 0.035, "(b)", fontsize=11.2, fontweight="bold", ha="center")
    save_figure(fig, "fig_experiment_testbed")


def plot_tier_differentiation() -> None:
    data = route_methods("evals/results/routeA_7uav_seed4111.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(1, 5)
    for method in METHOD_ORDER:
        vals = [data[method]["by_priority"][str(p)]["avg_latency_min"] for p in x]
        lw = 2.8 if method == "M1_ft" else 1.8
        alpha = 1.0 if method in {"M0c", "M1_sft", "M1_ft"} else 0.75
        ax1.plot(
            x,
            vals,
            marker="o",
            linewidth=lw,
            color=GG_COLORS[method],
            alpha=alpha,
            label=METHOD_LABELS[method],
        )
    ax1.set_xticks(x, [SERVICE_TIER_LABELS[i] for i in x])
    ax1.set_ylabel("Mean delivery time (min)")
    ax1.set_xlabel("Demand priority class")
    ax1.set_title("Delivery-time reallocation across priority classes")
    ax1.legend(ncol=2, frameon=True)
    style_axes(ax1)

    metrics = ["1st-priority\non-time", "2nd-priority\non-time", "3rd-priority\non-time", "4th-priority\non-time"]
    values = {
        method: [
            data[method]["by_priority"]["1"]["on_time_rate"] * 100,
            data[method]["by_priority"]["2"]["on_time_rate"] * 100,
            data[method]["by_priority"]["3"]["on_time_rate"] * 100,
            data[method]["by_priority"]["4"]["on_time_rate"] * 100,
        ]
        for method in METHOD_ORDER
    }
    width = 0.11
    xpos = np.arange(len(metrics))
    for i, method in enumerate(METHOD_ORDER):
        ax2.bar(
            xpos + (i - 3) * width,
            values[method],
            width=width,
            color=GG_COLORS[method],
            edgecolor="white",
            label=METHOD_LABELS[method],
        )
    ax2.set_ylim(40, 102)
    ax2.set_xticks(xpos, metrics)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("On-time rate by priority class")
    style_axes(ax2)

    fig.suptitle("Priority-class dispatch behavior under seven available UAVs", x=0.02, ha="left")
    save_figure(fig, "fig_tier_differentiation")


def plot_fleet_sensitivity() -> None:
    paths = {
        3: "evals/results/formal_comparison_seed4111.json",
        7: "evals/results/routeA_7uav_seed4111.json",
        10: "evals/results/routeA_10uav_seed4111.json",
    }
    rows = []
    for uav_supply, path in paths.items():
        data = route_methods(path)
        for method in METHOD_ORDER:
            if method in data:
                rows.append(
                    {
                        "uav_supply": uav_supply,
                        "method": method,
                        "first_priority_on": data[method]["by_priority"]["1"]["on_time_rate"] * 100,
                        "pw_on": data[method]["priority_weighted_on_time_score"] * 100,
                        "mean_delivery_time": data[method]["overall"]["avg_latency_min"],
                    }
                )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))
    panels = [
        ("first_priority_on", "First-priority demand on-time rate (%)", "First-priority demand timeliness"),
        ("pw_on", "Value-weighted on-time (%)", "Value-weighted service"),
        ("mean_delivery_time", "Mean delivery time (min)", "Overall delivery time"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for method in METHOD_ORDER:
            sub = df[df["method"] == method].sort_values("uav_supply")
            if sub.empty:
                continue
            ax.plot(
                sub["uav_supply"],
                sub[metric],
                marker="o",
                linewidth=2.8 if method == "M1_ft" else 1.8,
                color=GG_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=1.0 if method in {"M0c", "M1_sft", "M1_ft"} else 0.78,
            )
        ax.set_xticks([3, 7, 10])
        ax.set_xlabel("Available UAV supply")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", labelsize=8.8)
        style_axes(ax)
    legend_methods = ["M0c", "M0a", "M0b", "M1_pre", "M1_gemini", "M1_ft"]
    legend_labels = {
        "M0c": "Rule-based control",
        "M0a": "Uninformative control",
        "M0b": "No-priority control",
        "M1_pre": "Qwen3-4b",
        "M1_gemini": "gemini-3.1-flash-lite",
        "M1_ft": "Qwen3-4b-aligned (ours)",
    }
    handles = [
        mpl.lines.Line2D([0], [0], color=GG_COLORS[m], marker="o", linewidth=2, label=legend_labels[m])
        for m in legend_methods
    ]
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.31, wspace=0.28)
    legend = fig.legend(
        handles=handles,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.18),
        frameon=True,
        fontsize=8.0,
        handlelength=1.15,
        handletextpad=0.45,
        columnspacing=0.70,
        borderpad=0.32,
    )
    for text in legend.get_texts():
        if "(ours)" in text.get_text():
            text.set_fontweight("bold")
    fig.text(0.218, 0.070, "(a) First-priority demand timeliness", fontsize=11.2, ha="center")
    fig.text(0.535, 0.070, "(b) Value-weighted service", fontsize=11.2, ha="center")
    fig.text(0.850, 0.070, "(c) Overall delivery time", fontsize=11.2, ha="center")
    save_figure(fig, "fig_fleet_sensitivity")


def plot_priority_alignment() -> None:
    metrics = {
        "M1_pre": {
            "Exact tier acc.": 0.457,
            "Macro-F1": 0.402,
            "1st-priority precision": 0.270,
            "1st-priority recall": 0.500,
            "First-priority F1": 0.351,
            "Pairwise ordering": 0.745,
        },
        "M1_sft": {
            "Exact tier acc.": 0.576,
            "Macro-F1": 0.578,
            "1st-priority precision": 0.450,
            "1st-priority recall": 0.789,
            "First-priority F1": 0.573,
            "Pairwise ordering": 0.846,
        },
        "M1_ft": {
            "Exact tier acc.": 0.619,
            "Macro-F1": 0.620,
            "1st-priority precision": 0.523,
            "1st-priority recall": 0.790,
            "First-priority F1": 0.629,
            "Pairwise ordering": 0.844,
        },
        "M1_gemini": {
            "Exact tier acc.": 0.489,
            "Macro-F1": 0.473,
            "1st-priority precision": 0.300,
            "1st-priority recall": 0.895,
            "First-priority F1": 0.449,
            "Pairwise ordering": 0.863,
        },
    }
    bar_metrics = ["Exact tier acc.", "Macro-F1", "First-priority F1"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    xpos = np.arange(len(bar_metrics))
    width = 0.18
    plot_methods = ["M1_gemini", "M1_pre", "M1_sft", "M1_ft"]
    for i, method in enumerate(plot_methods):
        ax1.bar(
            xpos + (i - 1.5) * width,
            [metrics[method][m] for m in bar_metrics],
            width=width,
            color=GG_COLORS[method],
            edgecolor="white",
            label=METHOD_LABELS[method],
        )
    for j, metric in enumerate(bar_metrics):
        base_value = metrics["M1_pre"][metric]
        ax1.hlines(
            base_value,
            j - 0.42,
            j + 0.42,
            colors="#777777",
            linestyles="--",
            linewidth=0.8,
            alpha=0.45,
            label="Qwen3-4b baseline" if j == 0 else None,
        )
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(xpos, bar_metrics)
    ax1.set_ylabel("Score")
    ax1.legend(ncol=2, frameon=True, loc="upper left", fontsize=8.0)
    style_axes(ax1)

    for method in plot_methods:
        ax2.scatter(
            metrics[method]["1st-priority recall"],
            metrics[method]["1st-priority precision"],
            s=115,
            color=GG_COLORS[method],
            edgecolors="white",
            linewidths=1.0,
            label=METHOD_LABELS[method],
        )
    ax2.set_xlim(0.42, 0.94)
    ax2.set_ylim(0.22, 0.58)
    ax2.set_xlabel("First-priority demand recall")
    ax2.set_ylabel("First-priority demand precision")
    ax2.legend(
        frameon=True,
        fontsize=8.0,
        loc="upper left",
        markerscale=0.8,
        labelspacing=0.9,
        borderpad=0.75,
        handletextpad=0.9,
    )
    style_axes(ax2, grid_axis="both")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18, wspace=0.24)
    fig.text(0.32, 0.035, "(a) Priority-ranking quality", fontsize=11.2, ha="center")
    fig.text(0.78, 0.035, "(b) First-priority demand calibration", fontsize=11.2, ha="center")
    save_figure(fig, "fig_priority_alignment")


def _candidate_frame(method: str) -> pd.DataFrame:
    files = sorted((ROOT / f"data/eval_runs/routeA_7uav_{method.lower()}_seed4111").glob("*/nsga3_heuristic_results.json"))
    if not files:
        return pd.DataFrame()
    result = load_json(files[0].relative_to(ROOT))
    frontier_by_id = {row["solution_id"]: row for row in result["frontier"]}
    frontier_ids = set(frontier_by_id)
    rows = []
    for row in result["candidates"]:
        frontier_row = frontier_by_id.get(row["solution_id"], {})
        rows.append(
            {
                "method": method,
                "solution_id": row["solution_id"],
                "distance_km": row["final_total_distance_m"] / 1000.0,
                "avg_time_min": row["average_delivery_time_h"] * 60.0,
                "noise": row["final_total_noise_impact"],
                "activation_cost": row.get("drone_activation_cost"),
                "n_used_drones": row.get("n_used_drones"),
                "frontier": row["solution_id"] in frontier_ids,
                "frontier_result_path": frontier_row.get("frontier_result_path"),
            }
        )
    return pd.DataFrame(rows)


def _priority_manifest() -> dict[str, int]:
    return {
        row["event_id"]: int(row["latent_priority"])
        for row in load_jsonl("data/test/test_seeds/norm_eval/seed_4111/events_manifest.jsonl")
    }


def _frontier_service_metrics(path: str | None, manifest: dict[str, int]) -> dict[str, float | None]:
    if not path:
        return {"p1_on_time": None, "priority_weighted_on_time": None}
    result_path = Path(path)
    if not result_path.exists():
        result_path = ROOT / path
    if not result_path.exists():
        return {"p1_on_time": None, "priority_weighted_on_time": None}
    with open(result_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    p1_total = 0
    p1_on_time = 0
    urgent_total = 0
    urgent_on_time = 0
    weighted_total = 0.0
    weighted_on_time = 0.0
    for window in results:
        for demand in window.get("per_demand_results", []):
            event_id = demand.get("source_event_id") or demand.get("demand_id")
            priority = manifest.get(str(event_id), int(demand.get("priority", 4) or 4))
            on_time = bool(demand.get("is_deadline_met"))
            weight = PRIORITY_WEIGHTS.get(priority, 1.0)
            weighted_total += weight
            weighted_on_time += weight * float(on_time)
            if priority == 1:
                p1_total += 1
                p1_on_time += int(on_time)
            if priority <= 2:
                urgent_total += 1
                urgent_on_time += int(on_time)

    return {
        "p1_on_time": (100.0 * p1_on_time / p1_total) if p1_total else None,
        "urgent_on_time": (100.0 * urgent_on_time / urgent_total) if urgent_total else None,
        "priority_weighted_on_time": (100.0 * weighted_on_time / weighted_total) if weighted_total else None,
    }


def plot_pareto_selection() -> None:
    route = route_methods("evals/results/routeA_7uav_seed4111.json")
    rows = []
    for method in METHOD_ORDER:
        rows.append(
            {
                "method": method,
                "distance_1000km": route[method]["window_aggregate"]["total_distance_km"] / 1000.0,
                "noise_1000": route[method]["window_aggregate"]["total_noise_impact"] / 1000.0,
                "pw_on": route[method]["priority_weighted_on_time_score"] * 100.0,
            }
        )
    diag = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for _, row in diag.iterrows():
        method = row["method"]
        ax1.scatter(
            row["distance_1000km"],
            row["noise_1000"],
            s=55 + 2.5 * row["pw_on"],
            c=GG_COLORS[method],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
        ax1.text(row["distance_1000km"] + 0.3, row["noise_1000"], METHOD_LABELS[method], fontsize=7)
    ax1.set_xlim(diag["distance_1000km"].min() - 0.5, diag["distance_1000km"].max() + 2.1)
    ax1.set_xlabel("Aggregate distance (10^3 km)")
    ax1.set_ylabel("Route noise-exposure level (10^3 affected buildings)")
    ax1.set_title("Selected routes by dispatch policy")
    style_axes(ax1)

    cand = pd.concat([_candidate_frame("M0c"), _candidate_frame("M1_ft")], ignore_index=True)
    if not cand.empty:
        for method, sub in cand.groupby("method"):
            non_front = sub[~sub["frontier"]]
            front = sub[sub["frontier"]]
            ax2.scatter(
                non_front["distance_km"],
                non_front["avg_time_min"],
                c=GG_COLORS[method],
                s=28,
                alpha=0.25,
                edgecolors="none",
                label=f"{METHOD_LABELS[method]} candidates",
            )
            sc = ax2.scatter(
                front["distance_km"],
                front["avg_time_min"],
                c=front["noise"],
                cmap="viridis",
                s=58,
                marker="D" if method == "M1_ft" else "o",
                edgecolors=GG_COLORS[method],
                linewidths=1.0,
                label=f"{METHOD_LABELS[method]} frontier",
            )
        cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Route noise-exposure level")
    ax2.set_xlabel("Candidate distance (km)")
    ax2.set_ylabel("Average delivery time (min)")
    ax2.set_title("NSGA-III candidate frontier")
    ax2.legend(frameon=True, fontsize=7)
    style_axes(ax2)

    fig.suptitle("Multi-objective routing trade-offs", x=0.02, ha="left")
    save_figure(fig, "fig_pareto_selection")


def plot_pareto_frontier_p1_on_time() -> None:
    """NSGA-III frontier view colored by first- and second-priority on-time."""
    methods = ["M0c", "M1_ft"]
    labels = {
        "M0c": "Rule-based control",
        "M1_ft": "Qwen3-4b-aligned (ours)",
    }
    markers = {"M0c": "o", "M1_ft": "D"}
    cand = pd.concat([_candidate_frame(method) for method in methods], ignore_index=True)
    manifest = _priority_manifest()
    service_norm = mpl.colors.Normalize(vmin=93.0, vmax=99.0)
    service_cmap = mpl.colormaps["viridis"]
    candidate_color = "#D8D8D8"
    frontier_ranges = {}

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    sc = None
    for method in methods:
        sub = cand[cand["method"] == method]
        non_front = sub[~sub["frontier"]]
        front = sub[sub["frontier"]].copy()
        service_rows = [
            _frontier_service_metrics(path, manifest)
            for path in front["frontier_result_path"].tolist()
        ]
        front["top_two_on_time"] = [row["urgent_on_time"] for row in service_rows]
        front = front.dropna(subset=["top_two_on_time"])
        frontier_ranges[method] = (front["top_two_on_time"].min(), front["top_two_on_time"].max())

        ax.scatter(
            non_front["distance_km"],
            non_front["avg_time_min"],
            c=candidate_color,
            s=28,
            alpha=0.42,
            edgecolors="none",
            marker=markers[method],
            zorder=1,
        )
        marker_sizes = np.interp(front["top_two_on_time"], [93.0, 99.0], [58.0, 100.0])
        sc = ax.scatter(
            front["distance_km"],
            front["avg_time_min"],
            c=front["top_two_on_time"],
            cmap=service_cmap,
            norm=service_norm,
            s=marker_sizes,
            marker=markers[method],
            edgecolors=GG_COLORS[method],
            linewidths=1.25,
            alpha=0.98,
            zorder=4,
        )

    if sc is None:
        sc = mpl.cm.ScalarMappable(norm=service_norm, cmap=service_cmap)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("First- and second-priority demand on-time rate (%)")
    cbar.set_ticks([93, 95, 97, 99])
    ax.set_xlabel("Candidate distance (km)")
    ax.set_ylabel("Average delivery time (min)")
    ax.set_title("NSGA-III candidate frontier")
    if frontier_ranges:
        summary = "\n".join(
            f"{labels[method]}: {frontier_ranges[method][0]:.1f}-{frontier_ranges[method][1]:.1f}%"
            for method in methods
            if method in frontier_ranges
        )
        ax.text(
            0.02,
            0.97,
            "Frontier first- and second-priority on-time\n" + summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.86},
        )
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=candidate_color,
            markeredgecolor=candidate_color,
            linestyle="None",
            markersize=6,
            label="Non-frontier candidates",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker=markers["M0c"],
            color="none",
            markerfacecolor="white",
            markeredgecolor=GG_COLORS["M0c"],
            markeredgewidth=1.4,
            linestyle="None",
            markersize=7,
            label=f"{labels['M0c']} frontier",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker=markers["M1_ft"],
            color="none",
            markerfacecolor=service_cmap(service_norm(98.4)),
            markeredgecolor=GG_COLORS["M1_ft"],
            markeredgewidth=1.4,
            linestyle="None",
            markersize=7,
            label=f"{labels['M1_ft']} frontier",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    style_axes(ax)
    fig.suptitle(
        "NSGA-III frontier colored by first- and second-priority demand on-time rate",
        x=0.02,
        ha="left",
    )
    save_figure(fig, "fig_pareto_frontier_p1_on_time")


def _main_result_payload() -> dict[str, object]:
    data = route_methods("evals/results/routeA_7uav_seed4111.json")
    methods = ["M0a", "M0b", "M1_pre", "M1_gemini", "M1_ft"]
    labels = [method_label(m, width=20) for m in methods]
    return {
        "methods": methods,
        "labels": labels,
        "on_time": [
            (
                "First-priority demand",
                [data[m]["by_priority"]["1"]["on_time_rate"] * 100 for m in methods],
                "%",
                (35, 102),
                data["M0c"]["by_priority"]["1"]["on_time_rate"] * 100,
            ),
            (
                "Second-priority demand",
                [data[m]["by_priority"]["2"]["on_time_rate"] * 100 for m in methods],
                "%",
                (45, 102),
                data["M0c"]["by_priority"]["2"]["on_time_rate"] * 100,
            ),
        ],
        "delivery_time": [
            (
                "First-priority demand",
                [data[m]["by_priority"]["1"]["avg_latency_min"] for m in methods],
                " min",
                (0, 78),
                data["M0c"]["by_priority"]["1"]["avg_latency_min"],
            ),
            (
                "Second-priority demand",
                [data[m]["by_priority"]["2"]["avg_latency_min"] for m in methods],
                " min",
                (0, 78),
                data["M0c"]["by_priority"]["2"]["avg_latency_min"],
            ),
        ],
    }


def _annotate_bar_value(
    ax: mpl.axes.Axes,
    bar: mpl.patches.Rectangle,
    value: float,
    unit: str,
    xlim: tuple[float, float],
    fontsize: float = 6.5,
) -> None:
    near_right = value > xlim[1] - (9 if unit == "%" else 8)
    xpos = value + (0.8 if unit == "%" else 0.9)
    ha = "left"
    if near_right:
        xpos = min(value - (0.9 if unit == "%" else 1.0), xlim[1] - 1.0)
        ha = "right"
    ax.text(
        xpos,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}{unit}",
        va="center",
        ha=ha,
        fontsize=fontsize,
        color="#222222",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )


def _priority_comparison_handles() -> list[mpl.artist.Artist]:
    return [
        mpl.patches.Patch(facecolor="#D6D6D6", edgecolor="#666666", label="First-priority demand"),
        mpl.patches.Patch(facecolor="#D6D6D6", edgecolor="#666666", hatch="////", label="Second-priority demand"),
        mpl.lines.Line2D([0], [0], color="#4A4A4A", linestyle=(0, (5, 2)), linewidth=1.0, label="Rule-based, first-priority"),
        mpl.lines.Line2D([0], [0], color="#8A8A8A", linestyle=(0, (2, 2)), linewidth=1.0, label="Rule-based, second-priority"),
    ]


def _method_comparison_handles(methods: list[str], compact_labels: bool = False) -> list[mpl.artist.Artist]:
    compact = {
        "M0a": "Uninformative control",
        "M0b": "No-priority control",
        "M1_pre": "Qwen3-4b",
        "M1_gemini": "gemini-3.1-flash-lite",
        "M1_ft": "Qwen3-4b-aligned (ours)",
    }
    return [
        mpl.patches.Patch(
            facecolor=GG_COLORS[method],
            edgecolor="white",
            label=compact.get(method, METHOD_LABELS[method]) if compact_labels else METHOD_LABELS[method],
        )
        for method in methods
    ]


def _draw_grouped_priority_panel(
    ax: mpl.axes.Axes,
    metric_group: list[tuple[str, list[float], str, tuple[float, float], float]],
    methods: list[str],
    labels: list[str],
    title: str,
    xlabel: str,
    show_ylabels: bool = True,
    show_legend: bool = False,
) -> None:
    y = np.arange(len(methods))
    height = 0.29
    offsets = [-height / 1.7, height / 1.7]
    xlim = metric_group[0][3]
    for offset, (priority_label, values, unit, _, rule_value) in zip(offsets, metric_group):
        is_first_priority = priority_label.startswith("First")
        rule_color = "#4A4A4A" if is_first_priority else "#8A8A8A"
        rule_dash = (0, (5, 2)) if is_first_priority else (0, (2, 2))
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            color=[GG_COLORS[m] for m in methods],
            edgecolor="white",
            linewidth=0.8,
            hatch=None if is_first_priority else "////",
            alpha=0.96,
        )
        ax.axvline(rule_value, color=rule_color, linestyle=rule_dash, linewidth=1.0, alpha=0.78)
        for bar, value in zip(bars, values):
            _annotate_bar_value(ax, bar, value, unit, xlim, fontsize=6.4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_yticks(y, labels if show_ylabels else [])
    if not show_ylabels:
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    style_axes(ax, grid_axis="x")
    if show_ylabels:
        ax.set_ylabel("Dispatch policy")
    if show_legend:
        ax.legend(handles=_priority_comparison_handles(), frameon=True, loc="lower right", fontsize=6.1)


def _draw_single_priority_panel(
    ax: mpl.axes.Axes,
    metric: tuple[str, list[float], str, tuple[float, float], float],
    methods: list[str],
    labels: list[str],
    title: str,
    xlabel: str,
    show_ylabels: bool = False,
    show_rule_legend: bool = False,
) -> None:
    _, values, unit, xlim, rule_value = metric
    y = np.arange(len(methods))
    bars = ax.barh(
        y,
        values,
        height=0.68,
        color=[GG_COLORS[m] for m in methods],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.96,
    )
    ax.axvline(rule_value, color="#555555", linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.78, label="Rule-based control")
    for bar, value in zip(bars, values):
        _annotate_bar_value(ax, bar, value, unit, xlim, fontsize=7.1)
    ax.set_xlim(*xlim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    else:
        ax.set_xlabel("")
    ax.set_title(title, fontsize=10.2, pad=5)
    ax.set_yticks(y, labels if show_ylabels else [])
    if not show_ylabels:
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.invert_yaxis()
    style_axes(ax, grid_axis="x")
    if show_rule_legend:
        ax.legend(frameon=True, loc="lower right", fontsize=5.9)


def _draw_priority_dot_panel(
    ax: mpl.axes.Axes,
    metric_group: list[tuple[str, list[float], str, tuple[float, float], float]],
    methods: list[str],
    title: str,
    xlabel: str,
) -> None:
    row_positions = [1.0, 0.0]
    row_labels = ["1st-priority", "2nd-priority"]
    offsets = np.linspace(-0.21, 0.21, len(methods))
    xlim = metric_group[0][3]

    for row_y in row_positions:
        ax.axhspan(row_y - 0.32, row_y + 0.32, facecolor="white", alpha=0.28, zorder=0)

    for row_y, (_, values, unit, _, rule_value) in zip(row_positions, metric_group):
        ax.vlines(
            rule_value,
            row_y - 0.30,
            row_y + 0.30,
            color="#555555",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.82,
            zorder=2,
        )
        for offset, method, value in zip(offsets, methods, values):
            point_y = row_y + offset
            ax.scatter(
                value,
                point_y,
                s=38,
                color=GG_COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            near_right = value > xlim[1] - (8 if unit == "%" else 7)
            label_x = value + (0.7 if unit == "%" else 0.8)
            ha = "left"
            if near_right:
                label_x = min(value - (0.8 if unit == "%" else 0.9), xlim[1] - 0.9)
                ha = "right"
            ax.text(
                label_x,
                point_y,
                f"{value:.1f}{unit}",
                va="center",
                ha=ha,
                fontsize=6.0,
                color="#333333",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks(row_positions, row_labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10, pad=7)
    style_axes(ax, grid_axis="x")


def _pareto_frontier_handles() -> list[mpl.artist.Artist]:
    service_norm = mpl.colors.Normalize(vmin=93.0, vmax=99.0)
    service_cmap = mpl.colormaps["viridis"]
    return [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D8D8D8",
            markeredgecolor="#D8D8D8",
            linestyle="None",
            markersize=5.5,
            label="Non-frontier",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=GG_COLORS["M0c"],
            markeredgewidth=1.3,
            linestyle="None",
            markersize=6.2,
            label="Rule-based frontier",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=service_cmap(service_norm(98.4)),
            markeredgecolor=GG_COLORS["M1_ft"],
            markeredgewidth=1.3,
            linestyle="None",
            markersize=6.2,
            label="Qwen-aligned(ours) frontier",
        ),
    ]


def _draw_pareto_frontier_panel(
    ax: mpl.axes.Axes,
    fig: mpl.figure.Figure,
    title: str,
    legend_y: float = -0.18,
    compact: bool = False,
    show_legend: bool = True,
) -> None:
    methods = ["M0c", "M1_ft"]
    labels = {"M0c": "Rule-based control", "M1_ft": "Qwen3-4b-aligned (ours)"}
    markers = {"M0c": "o", "M1_ft": "D"}
    cand = pd.concat([_candidate_frame(method) for method in methods], ignore_index=True)
    manifest = _priority_manifest()
    service_norm = mpl.colors.Normalize(vmin=93.0, vmax=99.0)
    service_cmap = mpl.colormaps["viridis"]
    candidate_color = "#D8D8D8"
    frontier_ranges = {}
    sc = None

    for method in methods:
        sub = cand[cand["method"] == method]
        non_front = sub[~sub["frontier"]]
        front = sub[sub["frontier"]].copy()
        service_rows = [_frontier_service_metrics(path, manifest) for path in front["frontier_result_path"].tolist()]
        front["top_two_on_time"] = [row["urgent_on_time"] for row in service_rows]
        front = front.dropna(subset=["top_two_on_time"])
        frontier_ranges[method] = (front["top_two_on_time"].min(), front["top_two_on_time"].max())

        ax.scatter(
            non_front["distance_km"],
            non_front["avg_time_min"],
            c=candidate_color,
            s=18 if compact else 24,
            alpha=0.36,
            edgecolors="none",
            marker=markers[method],
            zorder=1,
        )
        marker_sizes = np.interp(front["top_two_on_time"], [93.0, 99.0], [46.0, 82.0] if compact else [54.0, 94.0])
        sc = ax.scatter(
            front["distance_km"],
            front["avg_time_min"],
            c=front["top_two_on_time"],
            cmap=service_cmap,
            norm=service_norm,
            s=marker_sizes,
            marker=markers[method],
            edgecolors=GG_COLORS[method],
            linewidths=1.15,
            alpha=0.98,
            zorder=4,
        )

    if sc is None:
        sc = mpl.cm.ScalarMappable(norm=service_norm, cmap=service_cmap)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.050 if compact else 0.046, pad=0.025)
    cbar.set_label("First- and second-priority on-time rate (%)", fontsize=7.2 if compact else 9.0)
    cbar.set_ticks([93, 95, 97, 99])
    cbar.ax.tick_params(labelsize=6.7 if compact else 8.4)
    ax.set_xlabel("Candidate distance (km)", fontsize=8.0 if compact else 9.5)
    ax.set_ylabel("Average delivery time (min)", fontsize=8.0 if compact else 9.5)
    ax.tick_params(axis="both", labelsize=7.2 if compact else 8.7)
    ax.set_title(title)

    if frontier_ranges:
        summary = "\n".join(
            f"{labels[method]}: {frontier_ranges[method][0]:.1f}-{frontier_ranges[method][1]:.1f}%"
            for method in methods
        )
        ax.text(
            0.02,
            0.97,
            "Frontier on-time range\n" + summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4 if compact else 7.5,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.86},
        )

    if show_legend:
        ax.legend(
            handles=_pareto_frontier_handles(),
            frameon=True,
            fontsize=6.1 if compact else 6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=1 if compact else 3,
            handletextpad=0.5,
            columnspacing=0.8,
        )
    style_axes(ax)


def plot_main_pareto_combo_abc() -> None:
    """Candidate combined figure: three horizontal panels."""
    payload = _main_result_payload()
    methods = payload["methods"]
    labels = payload["labels"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.8, 4.8),
        gridspec_kw={"width_ratios": [1.18, 1.10, 1.28], "wspace": 0.28},
    )
    _draw_grouped_priority_panel(
        axes[0],
        payload["on_time"],
        methods,
        labels,
        "(a) High-priority on-time rate",
        "On-time rate (%)",
        show_ylabels=True,
        show_legend=False,
    )
    _draw_grouped_priority_panel(
        axes[1],
        payload["delivery_time"],
        methods,
        labels,
        "(b) High-priority delivery time",
        "Delivery time (min)",
        show_ylabels=False,
    )
    _draw_pareto_frontier_panel(
        axes[2],
        fig,
        "(c) Routing frontier by service quality",
        legend_y=-0.18,
        compact=True,
    )
    fig.legend(
        handles=_priority_comparison_handles(),
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.35, 0.90),
        ncol=4,
        fontsize=6.6,
        handletextpad=0.6,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.25)
    fig.suptitle("Main service effect and Pareto-frontier routing trade-off", x=0.02, ha="left")
    save_figure(fig, "fig_main_pareto_combo_abc")


def plot_main_pareto_combo_quad() -> None:
    """Candidate combined figure: four metric panels plus one frontier panel."""
    payload = _main_result_payload()
    methods = payload["methods"]
    labels = payload["labels"]

    fig = plt.figure(figsize=(13.4, 6.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.16)
    left = outer[0, 0].subgridspec(2, 2, hspace=0.46, wspace=0.14)
    axes = [
        fig.add_subplot(left[0, 0]),
        fig.add_subplot(left[0, 1]),
        fig.add_subplot(left[1, 0]),
        fig.add_subplot(left[1, 1]),
    ]
    pareto_ax = fig.add_subplot(outer[0, 1])

    single_metrics = [
        (payload["on_time"][0], "1st-priority on-time rate", "", False, False),
        (payload["on_time"][1], "2nd-priority on-time rate", "", False, False),
        (payload["delivery_time"][0], "1st-priority delivery time", "Delivery time (min)", False, False),
        (payload["delivery_time"][1], "2nd-priority delivery time", "Delivery time (min)", False, False),
    ]
    for ax, (metric, title, xlabel, show_ylabels, show_rule_legend) in zip(axes, single_metrics):
        _draw_single_priority_panel(
            ax,
            metric,
            methods,
            labels,
            title,
            xlabel,
            show_ylabels=show_ylabels,
            show_rule_legend=show_rule_legend,
        )
    _draw_pareto_frontier_panel(
        pareto_ax,
        fig,
        "",
        legend_y=-0.18,
        compact=False,
        show_legend=False,
    )
    pareto_ax.set_box_aspect(1)
    legend_handles = [
        mpl.lines.Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), linewidth=1.2, label="Rule-based control")
    ] + _method_comparison_handles(methods, compact_labels=True)
    left_legend = fig.legend(
        handles=legend_handles,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.335, 0.20),
        ncol=6,
        fontsize=8.0,
        handlelength=1.25,
        handletextpad=0.42,
        columnspacing=0.70,
        borderpad=0.28,
        labelspacing=0.35,
    )
    right_legend = fig.legend(
        handles=_pareto_frontier_handles(),
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.812, 0.20),
        ncol=3,
        fontsize=8.0,
        handlelength=1.15,
        handletextpad=0.45,
        columnspacing=0.75,
        borderpad=0.30,
    )
    for legend in (left_legend, right_legend):
        for text in legend.get_texts():
            if "(ours)" in text.get_text() or "Aligned-Qwen" in text.get_text():
                text.set_fontweight("bold")
    fig.text(0.335, 0.12, "(a) Priority-specific service outcomes", fontsize=11.2, ha="center")
    fig.text(0.812, 0.12, "(b) Routing frontier by service quality", fontsize=11.2, ha="center")
    fig.subplots_adjust(left=0.040, right=0.98, top=0.96, bottom=0.28)
    save_figure(fig, "fig_main_pareto_combo_quad")


def plot_main_pareto_combo_balanced() -> None:
    """Cleaner candidate: compact outcome dot panels plus Pareto frontier."""
    payload = _main_result_payload()
    methods = payload["methods"]

    fig = plt.figure(figsize=(14.6, 6.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.20)
    left = outer[0, 0].subgridspec(2, 1, hspace=0.55)
    ax_on = fig.add_subplot(left[0, 0])
    ax_time = fig.add_subplot(left[1, 0])
    pareto_ax = fig.add_subplot(outer[0, 1])

    _draw_priority_dot_panel(
        ax_on,
        payload["on_time"],
        methods,
        "On-time rate",
        "On-time rate (%)",
    )
    _draw_priority_dot_panel(
        ax_time,
        payload["delivery_time"],
        methods,
        "Delivery time",
        "Delivery time (min)",
    )
    _draw_pareto_frontier_panel(
        pareto_ax,
        fig,
        "(b) Routing frontier by service quality",
        legend_y=-0.17,
        compact=False,
    )

    legend_handles = _method_comparison_handles(methods) + [
        mpl.lines.Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), linewidth=1.1, label="Rule-based control")
    ]
    fig.legend(
        handles=legend_handles,
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.045, 0.035),
        ncol=3,
        fontsize=6.8,
        handletextpad=0.55,
        columnspacing=0.95,
    )
    fig.text(0.02, 0.82, "(a) Priority-specific service outcomes", fontsize=11, fontweight="bold", ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.24)
    fig.suptitle("Main service effect and Pareto-frontier routing trade-off", x=0.02, ha="left")
    save_figure(fig, "fig_main_pareto_combo_balanced")


def plot_pareto_frontier_p1_on_time_extended() -> None:
    """Expanded NSGA-III frontier view with additional baselines and LLM variants."""
    methods = ["M0a", "M0b", "M0c", "M1_pre", "M1_gemini", "M1_sft", "M1_ft"]
    labels = {
        "M0a": "Random",
        "M0b": "Uniform",
        "M0c": "Rule",
        "M1_pre": "Qwen base",
        "M1_gemini": "Gemini",
        "M1_sft": "SFT (ours)",
        "M1_ft": "SFT+GRPO (ours)",
    }
    markers = {
        "M0a": "o",
        "M0b": "s",
        "M0c": "^",
        "M1_pre": "v",
        "M1_gemini": "P",
        "M1_sft": "D",
        "M1_ft": "*",
    }
    cand = pd.concat([_candidate_frame(method) for method in methods], ignore_index=True)
    manifest = _priority_manifest()
    p1_norm = mpl.colors.Normalize(vmin=38.0, vmax=100.0)
    p1_cmap = mpl.colormaps["viridis"]
    candidate_color = "#D8D8D8"

    fig, ax = plt.subplots(figsize=(8.9, 5.2))
    for method in methods:
        sub = cand[cand["method"] == method]
        non_front = sub[~sub["frontier"]]
        front = sub[sub["frontier"]].copy()
        service_rows = [
            _frontier_service_metrics(path, manifest)
            for path in front["frontier_result_path"].tolist()
        ]
        front["p1_on_time"] = [row["p1_on_time"] for row in service_rows]
        front = front.dropna(subset=["p1_on_time"])

        ax.scatter(
            non_front["distance_km"],
            non_front["avg_time_min"],
            c=candidate_color,
            s=20,
            alpha=0.28,
            edgecolors="none",
            marker=markers[method],
            zorder=1,
        )
        size = 132 if method == "M1_ft" else 86 if method == "M1_sft" else 68
        sc = ax.scatter(
            front["distance_km"],
            front["avg_time_min"],
            c=front["p1_on_time"],
            cmap=p1_cmap,
            norm=p1_norm,
            s=size,
            marker=markers[method],
            edgecolors=GG_COLORS[method],
            linewidths=1.6 if method in {"M1_sft", "M1_ft"} else 1.1,
            alpha=0.98,
            zorder=4 if method in {"M1_sft", "M1_ft"} else 3,
        )

    method_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=candidate_color,
            markeredgecolor=candidate_color,
            linestyle="None",
            markersize=6,
            label="Non-frontier candidates",
        ),
    ] + [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=markers[method],
            color="none",
            markerfacecolor="white",
            markeredgecolor=GG_COLORS[method],
            markeredgewidth=1.4,
            linestyle="None",
            markersize=8.5 if method == "M1_ft" else 7,
            label=labels[method],
        )
        for method in methods
    ]
    ax.legend(
        handles=method_handles,
        frameon=True,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Frontier 1st-priority on-time rate (%)")
    cbar.set_ticks([40, 60, 80, 90, 95, 100])
    ax.set_xlabel("Candidate distance (km)")
    ax.set_ylabel("Average delivery time (min)")
    ax.set_title("NSGA-III candidate frontier")
    style_axes(ax)
    fig.suptitle("Expanded frontier view: all baselines and LLM variants", x=0.02, ha="left")
    save_figure(fig, "fig_pareto_frontier_p1_on_time_extended")


def plot_pareto_frontier_ours() -> None:
    """Pareto-style view focused on rule, SFT, and SFT+GRPO frontiers."""
    route = route_methods("evals/results/routeA_7uav_seed4111.json")
    methods = ["M0c", "M1_sft", "M1_ft"]
    labels = {
        "M0c": "Rule heuristic",
        "M1_sft": "SFT (ours)",
        "M1_ft": "SFT+GRPO (ours)",
    }
    markers = {"M0c": "o", "M1_sft": "D", "M1_ft": "*"}
    selected = []
    for method in methods:
        data = route[method]
        selected.append(
            {
                "method": method,
                "distance_1000km": data["window_aggregate"]["total_distance_km"] / 1000.0,
                "noise_1000": data["window_aggregate"]["total_noise_impact"] / 1000.0,
                "p1_on_time": data["by_priority"]["1"]["on_time_rate"] * 100.0,
            }
        )
    selected_df = pd.DataFrame(selected)

    cand = pd.concat([_candidate_frame(method) for method in methods], ignore_index=True)
    frontier = cand[cand["frontier"]].copy()
    manifest = _priority_manifest()
    service_rows = [
        _frontier_service_metrics(path, manifest)
        for path in frontier["frontier_result_path"].tolist()
    ]
    frontier["p1_on_time"] = [row["p1_on_time"] for row in service_rows]
    frontier["priority_weighted_on_time"] = [row["priority_weighted_on_time"] for row in service_rows]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.9),
        gridspec_kw={"width_ratios": [0.92, 1.25]},
    )

    for _, row in selected_df.iterrows():
        method = row["method"]
        is_ours = method != "M0c"
        ax1.scatter(
            row["distance_1000km"],
            row["noise_1000"],
            s=145 if method == "M1_sft" else 205 if method == "M1_ft" else 105,
            marker=markers[method],
            c=GG_COLORS[method],
            edgecolors="white",
            linewidths=1.2,
            alpha=0.94,
            zorder=3,
        )
        ax1.text(
            row["distance_1000km"] + 0.12,
            row["noise_1000"] + (0.6 if method == "M0c" else 0.9),
            f"{labels[method]}\nP1 on-time {row['p1_on_time']:.1f}%",
            fontsize=7.5,
            fontweight="bold" if is_ours else "normal",
            color="#222222" if is_ours else "#555555",
        )
    rule_row = selected_df[selected_df["method"] == "M0c"].iloc[0]
    ax1.axvline(rule_row["distance_1000km"], color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.axhline(rule_row["noise_1000"], color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_xlabel("Aggregate distance (10^3 km)")
    ax1.set_ylabel("Route noise-exposure level (10^3 affected buildings)")
    ax1.set_title("Selected dispatch outcome")
    ax1.set_xlim(selected_df["distance_1000km"].min() - 0.9, selected_df["distance_1000km"].max() + 1.6)
    ax1.set_ylim(selected_df["noise_1000"].min() - 4.5, selected_df["noise_1000"].max() + 6.5)
    style_axes(ax1)

    noise_norm = mpl.colors.Normalize(vmin=float(cand["noise"].min()), vmax=float(cand["noise"].max()))
    cmap = mpl.colormaps["viridis"]
    for method in methods:
        sub = cand[cand["method"] == method]
        front = frontier[frontier["method"] == method].copy()
        front_unique = (
            front.sort_values(["distance_km", "avg_time_min", "noise"])
            .drop_duplicates(["distance_km", "avg_time_min", "noise"])
        )
        ax2.scatter(
            sub["distance_km"],
            sub["avg_time_min"],
            s=24,
            marker=markers[method],
            c=GG_COLORS[method],
            alpha=0.14 if method == "M0c" else 0.18,
            edgecolors="none",
            zorder=1,
        )
        p1 = front_unique["p1_on_time"].fillna(90.0)
        sizes = 54 + np.clip(p1 - 84.0, 0.0, 16.0) * (8.0 if method == "M1_ft" else 6.5)
        ax2.scatter(
            front_unique["distance_km"],
            front_unique["avg_time_min"],
            c=front_unique["noise"],
            cmap=cmap,
            norm=noise_norm,
            s=sizes,
            marker=markers[method],
            edgecolors=GG_COLORS[method],
            linewidths=1.0 if method == "M0c" else 1.7,
            alpha=0.98,
            zorder=4 if method != "M0c" else 3,
        )

    best_rows = (
        frontier.sort_values(["method", "p1_on_time", "avg_time_min"], ascending=[True, False, True])
        .drop_duplicates("method")
        .set_index("method")
    )
    for method, dx, dy in [
        ("M0c", -24, 0.26),
        ("M1_sft", 8, -0.18),
        ("M1_ft", 8, 0.22),
    ]:
        if method not in best_rows.index:
            continue
        row = best_rows.loc[method]
        ax2.text(
            row["distance_km"] + dx,
            row["avg_time_min"] + dy,
            f"{labels[method]}\nP1 {row['p1_on_time']:.1f}%",
            fontsize=7.2,
            fontweight="bold" if method != "M0c" else "normal",
            color="#222222" if method != "M0c" else "#555555",
        )

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=GG_COLORS[method],
            marker=markers[method],
            markerfacecolor="white",
            markeredgecolor=GG_COLORS[method],
            linestyle="None",
            markersize=7 if method != "M1_ft" else 9,
            label=labels[method],
        )
        for method in methods
    ]
    ax2.legend(handles=handles, frameon=True, fontsize=7, loc="upper right")
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=noise_norm, cmap=cmap),
        ax=ax2,
        fraction=0.046,
        pad=0.035,
    )
    cbar.set_label("Route noise-exposure level")
    ax2.set_xlabel("Candidate distance (km)")
    ax2.set_ylabel("Average delivery time (min)")
    ax2.set_title("NSGA-III candidate frontier")
    ax2.set_ylim(float(cand["avg_time_min"].min()) - 0.35, float(cand["avg_time_min"].max()) + 0.2)
    style_axes(ax2)

    fig.suptitle("Pareto candidates show where aligned Qwen policies gain critical service", x=0.02, ha="left")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.16, wspace=0.24)
    save_figure(fig, "fig_pareto_frontier_ours")


def plot_priority_aware_pareto_filtering() -> None:
    """Filter four-objective Pareto candidates by a critical-service SLA."""
    methods = ["M0c", "M1_sft", "M1_ft"]
    labels = {
        "M0c": "Rule heuristic",
        "M1_sft": "SFT (ours)",
        "M1_ft": "SFT+GRPO (ours)",
    }
    markers = {"M0c": "o", "M1_sft": "D", "M1_ft": "*"}
    sla_threshold = 95.0

    cand = pd.concat([_candidate_frame(method) for method in methods], ignore_index=True)
    frontier = cand[cand["frontier"]].copy()
    manifest = _priority_manifest()
    service_rows = [
        _frontier_service_metrics(path, manifest)
        for path in frontier["frontier_result_path"].tolist()
    ]
    frontier["p1_on_time"] = [row["p1_on_time"] for row in service_rows]
    frontier["urgent_on_time"] = [row["urgent_on_time"] for row in service_rows]
    frontier["priority_weighted_on_time"] = [row["priority_weighted_on_time"] for row in service_rows]
    frontier["sla_pass"] = frontier["p1_on_time"] >= sla_threshold

    n_used_values = sorted(cand["n_used_drones"].dropna().unique().tolist())
    if len(n_used_values) > 1:
        size_metric = "n_used_drones"
        size_label = "Used UAVs"
    else:
        size_metric = "activation_cost"
        size_label = "UAV activation-cost setting"

    size_values = cand[size_metric].astype(float)
    size_min = float(size_values.min())
    size_span = float(size_values.max() - size_min) or 1.0

    def marker_size(value: object, low: float = 36.0, high: float = 190.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = size_min
        return low + (numeric - size_min) / size_span * (high - low)

    pass_counts = {
        method: (
            int(frontier[(frontier["method"] == method) & (frontier["sla_pass"])].shape[0]),
            int(frontier[frontier["method"] == method].shape[0]),
        )
        for method in methods
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.95),
        gridspec_kw={"width_ratios": [1.28, 0.82]},
    )
    noise_norm = mpl.colors.Normalize(vmin=float(cand["noise"].min()), vmax=float(cand["noise"].max()))
    cmap = mpl.colormaps["viridis"]

    for method in methods:
        sub = cand[cand["method"] == method]
        front = frontier[frontier["method"] == method].copy()
        fail = front[~front["sla_pass"]]
        passed = front[front["sla_pass"]]

        ax1.scatter(
            sub["distance_km"],
            sub["avg_time_min"],
            c=sub["noise"],
            cmap=cmap,
            norm=noise_norm,
            s=22,
            marker=markers[method],
            alpha=0.11,
            edgecolors="none",
            zorder=1,
        )
        if not fail.empty:
            ax1.scatter(
                fail["distance_km"],
                fail["avg_time_min"],
                facecolors="#D7D7D7",
                s=[marker_size(v) for v in fail[size_metric]],
                marker=markers[method],
                edgecolors="#8A8A8A",
                linewidths=0.75,
                alpha=0.72,
                zorder=2,
            )
        if not passed.empty:
            ax1.scatter(
                passed["distance_km"],
                passed["avg_time_min"],
                c=passed["noise"],
                cmap=cmap,
                norm=noise_norm,
                s=[marker_size(v, low=72.0, high=250.0) for v in passed[size_metric]],
                marker=markers[method],
                edgecolors=GG_COLORS[method],
                linewidths=1.7,
                alpha=0.98,
                zorder=4,
            )

    feasible = frontier[frontier["sla_pass"]]
    if not feasible.empty:
        best = feasible.sort_values(["avg_time_min", "distance_km"]).iloc[0]
        ax1.text(
            best["distance_km"] + 9,
            best["avg_time_min"] - 0.12,
            "SLA-feasible\nfrontier",
            fontsize=7.5,
            color="#222222",
            fontweight="bold",
        )

    method_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="None",
            marker=markers[method],
            markerfacecolor="white",
            markeredgecolor=GG_COLORS[method],
            markeredgewidth=1.5,
            markersize=7 if method != "M1_ft" else 9,
            label=labels[method],
        )
        for method in methods
    ]
    filter_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="#444444",
            markeredgecolor="#111111",
            markersize=7,
            label=f"P1 SLA >= {sla_threshold:.0f}%",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="#DDDDDD",
            markeredgecolor="#888888",
            markersize=7,
            label="Filtered out",
        ),
    ]
    legend1 = ax1.legend(handles=method_handles, frameon=True, fontsize=7, loc="upper right")
    ax1.add_artist(legend1)
    ax1.legend(handles=filter_handles, frameon=True, fontsize=7, loc="lower right")

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=noise_norm, cmap=cmap),
        ax=ax1,
        fraction=0.046,
        pad=0.025,
    )
    cbar.set_label("Route noise-exposure level")

    ax1.set_xlabel("Candidate distance (km)")
    ax1.set_ylabel("Average delivery time (min)")
    ax1.set_title("Four-objective candidates after priority-aware filtering")
    count_text = " | ".join(
        f"{labels[method]} {passed}/{total}"
        for method, (passed, total) in pass_counts.items()
    )
    ax1.text(
        0.01,
        0.985,
        f"SLA-passing frontier points: {count_text}",
        transform=ax1.transAxes,
        fontsize=7.2,
        color="#333333",
        va="top",
    )
    style_axes(ax1)

    x_positions = {method: i for i, method in enumerate(methods)}
    for method in methods:
        front = frontier[frontier["method"] == method].copy().sort_values(["p1_on_time", "distance_km"])
        offsets = np.linspace(-0.17, 0.17, len(front)) if len(front) > 1 else np.array([0.0])
        for offset, (_, row) in zip(offsets, front.iterrows()):
            passed = bool(row["sla_pass"])
            ax2.scatter(
                x_positions[method] + offset,
                row["p1_on_time"],
                s=marker_size(row[size_metric], low=36.0, high=180.0),
                marker=markers[method],
                facecolors=GG_COLORS[method] if passed else "none",
                edgecolors=GG_COLORS[method] if passed else "#8A8A8A",
                linewidths=1.2 if passed else 0.8,
                alpha=0.9 if passed else 0.55,
            )
        passed_count, total = pass_counts[method]
        y_top = max(99.6, float(front["p1_on_time"].max()) + 0.35)
        ax2.text(
            x_positions[method],
            y_top,
            f"{passed_count}/{total}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if method != "M0c" else "normal",
            color=GG_COLORS[method],
        )

    ax2.axhline(sla_threshold, color="#333333", linestyle="--", linewidth=1.0, alpha=0.85)
    ax2.text(
        -0.43,
        sla_threshold + 0.25,
        f"P1 SLA {sla_threshold:.0f}%",
        fontsize=7.5,
        color="#333333",
        ha="left",
    )
    ax2.set_xticks([x_positions[method] for method in methods], [labels[method] for method in methods], rotation=15)
    ax2.set_ylabel("P1 on-time among frontier plans (%)")
    ax2.set_ylim(84.0, 100.6)
    ax2.set_xlim(-0.5, len(methods) - 0.5)
    ax2.set_title("Critical-demand SLA retention")
    size_note = (
        f"Marker size: {size_label}."
        if len(n_used_values) > 1
        else f"All plans use {n_used_values[0]:.0f} UAVs; marker size shows activation-cost setting."
    )
    ax2.text(
        0.02,
        0.03,
        size_note,
        transform=ax2.transAxes,
        fontsize=7.1,
        color="#555555",
        ha="left",
    )
    style_axes(ax2)

    fig.suptitle("Priority-aware Pareto filtering keeps only cost trade-offs that meet critical-service SLA", x=0.02, ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.18, wspace=0.22)
    save_figure(fig, "fig_priority_aware_pareto_filtering")


def plot_priority_constrained_frontier() -> None:
    """Show contribution as SLA-constrained dominance over routing-cost frontiers."""
    methods = ["M0c", "M1_sft", "M1_ft"]
    labels = {
        "M0c": "Rule heuristic",
        "M1_sft": "SFT (ours)",
        "M1_ft": "SFT+GRPO (ours)",
    }
    markers = {"M0c": "o", "M1_sft": "D", "M1_ft": "*"}
    sla_threshold = 95.0

    manifest = _priority_manifest()
    frames = []
    for method in methods:
        frame = _candidate_frame(method)
        front = frame[frame["frontier"]].copy()
        service_rows = [
            _frontier_service_metrics(path, manifest)
            for path in front["frontier_result_path"].tolist()
        ]
        front["p1_on_time"] = [row["p1_on_time"] for row in service_rows]
        front["priority_weighted_on_time"] = [row["priority_weighted_on_time"] for row in service_rows]
        frames.append(front)
    frontier = pd.concat(frames, ignore_index=True)

    varied_cost_metrics = ["distance_km", "avg_time_min", "noise"]
    for metric in varied_cost_metrics:
        span = float(frontier[metric].max() - frontier[metric].min()) or 1.0
        frontier[f"{metric}_norm"] = (frontier[metric] - float(frontier[metric].min())) / span
    frontier["route_cost_index"] = 100.0 * frontier[
        [f"{metric}_norm" for metric in varied_cost_metrics]
    ].mean(axis=1)
    frontier["sla_pass"] = frontier["p1_on_time"] >= sla_threshold

    pass_counts = {
        method: (
            int(frontier[(frontier["method"] == method) & (frontier["sla_pass"])].shape[0]),
            int(frontier[frontier["method"] == method].shape[0]),
        )
        for method in methods
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.8),
        gridspec_kw={"width_ratios": [1.35, 0.75]},
    )

    ax1.axvspan(sla_threshold, 100.4, color="#DCEEFF", alpha=0.45, zorder=0)
    ax1.axvline(sla_threshold, color="#444444", linestyle="--", linewidth=1.0, alpha=0.9)
    ax1.text(
        sla_threshold + 0.12,
        0.97,
        "Priority-feasible region",
        transform=ax1.get_xaxis_transform(),
        fontsize=8,
        color="#28527A",
        va="top",
    )
    for method in methods:
        sub = frontier[frontier["method"] == method].copy()
        fail = sub[~sub["sla_pass"]]
        passed = sub[sub["sla_pass"]]
        if not fail.empty:
            ax1.scatter(
                fail["p1_on_time"],
                fail["route_cost_index"],
                s=58,
                marker=markers[method],
                facecolors="none",
                edgecolors="#9A9A9A",
                linewidths=1.0,
                alpha=0.75,
                zorder=2,
            )
        if not passed.empty:
            ax1.scatter(
                passed["p1_on_time"],
                passed["route_cost_index"],
                s=82 if method != "M1_ft" else 130,
                marker=markers[method],
                facecolors=GG_COLORS[method],
                edgecolors="white",
                linewidths=0.9,
                alpha=0.92,
                zorder=3,
            )

    best_feasible = (
        frontier[frontier["sla_pass"]]
        .sort_values(["route_cost_index", "p1_on_time"], ascending=[True, False])
        .drop_duplicates("method")
    )
    for _, row in best_feasible.iterrows():
        method = row["method"]
        ax1.text(
            row["p1_on_time"] + (0.15 if method != "M1_ft" else -1.35),
            row["route_cost_index"] + (1.5 if method != "M1_ft" else -4.5),
            f"{labels[method]}\nbest feasible cost {row['route_cost_index']:.1f}",
            fontsize=7.2,
            fontweight="bold" if method != "M0c" else "normal",
            color="#222222",
        )

    rule = frontier[frontier["method"] == "M0c"]
    ax1.text(
        float(rule["p1_on_time"].max()) - 1.1,
        float(rule["route_cost_index"].min()) - 4.0,
        "Rule frontier stays\nbelow P1 SLA",
        fontsize=7.2,
        color="#555555",
        ha="right",
    )

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="None",
            marker=markers[method],
            markerfacecolor=GG_COLORS[method],
            markeredgecolor="white",
            markersize=7 if method != "M1_ft" else 9,
            label=labels[method],
        )
        for method in methods
    ]
    ax1.legend(handles=handles, frameon=True, fontsize=7, loc="upper left")
    ax1.set_xlabel("Critical-demand service: P1 on-time (%)")
    ax1.set_ylabel("Normalized route-cost index\n(distance + time + noise, lower is better)")
    ax1.set_xlim(84.2, 100.1)
    ax1.set_ylim(-3.0, max(82.0, float(frontier["route_cost_index"].max()) + 5.0))
    ax1.set_title("Frontier cost vs critical-service SLA")
    ax1.text(
        0.01,
        0.02,
        "Service-rate loss = 0 and used UAVs = 7 for all frontier points in this run;\n"
        "the plotted cost index therefore uses the three non-constant route objectives.",
        transform=ax1.transAxes,
        fontsize=7.0,
        color="#555555",
        ha="left",
        va="bottom",
    )
    style_axes(ax1, grid_axis="both")

    y = np.arange(len(methods))
    rates = []
    colors = []
    for method in methods:
        passed, total = pass_counts[method]
        rates.append(100.0 * passed / total if total else 0.0)
        colors.append(GG_COLORS[method])
    bars = ax2.barh(y, rates, color=colors, edgecolor="white", height=0.62)
    ax2.axvline(100.0, color="#333333", linewidth=0.8, alpha=0.3)
    for bar, method, rate in zip(bars, methods, rates):
        passed, total = pass_counts[method]
        ax2.text(
            min(rate + 3.0, 96.0),
            bar.get_y() + bar.get_height() / 2,
            f"{passed}/{total}",
            va="center",
            ha="left" if rate < 92 else "right",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )
    ax2.set_yticks(y, [labels[method] for method in methods])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Frontier plans meeting P1 SLA (%)")
    ax2.set_title("Retention after SLA filter")
    style_axes(ax2, grid_axis="x")

    fig.suptitle("Priority-constrained Pareto frontier: critical-service feasibility first, routing cost second", x=0.02, ha="left")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.20, wspace=0.26)
    save_figure(fig, "fig_priority_constrained_frontier")


def plot_priority_aware_performance_frontier() -> None:
    """Single-panel contribution view: critical service versus route cost."""
    methods = ["M0c", "M1_sft", "M1_ft"]
    labels = {
        "M0c": "Rule heuristic",
        "M1_sft": "SFT (ours)",
        "M1_ft": "SFT+GRPO (ours)",
    }
    markers = {"M0c": "o", "M1_sft": "D", "M1_ft": "*"}

    manifest = _priority_manifest()
    frames = []
    for method in methods:
        frame = _candidate_frame(method)
        front = frame[frame["frontier"]].copy()
        service_rows = [
            _frontier_service_metrics(path, manifest)
            for path in front["frontier_result_path"].tolist()
        ]
        front["p1_on_time"] = [row["p1_on_time"] for row in service_rows]
        frames.append(front)
    frontier = pd.concat(frames, ignore_index=True)

    cost_metrics = ["distance_km", "avg_time_min", "noise"]
    for metric in cost_metrics:
        span = float(frontier[metric].max() - frontier[metric].min()) or 1.0
        frontier[f"{metric}_norm"] = (frontier[metric] - float(frontier[metric].min())) / span
    frontier["route_cost_index"] = 100.0 * frontier[
        [f"{metric}_norm" for metric in cost_metrics]
    ].mean(axis=1)

    nondominated = []
    for idx, candidate in frontier.iterrows():
        dominated = False
        for other_idx, other in frontier.iterrows():
            if idx == other_idx:
                continue
            better_or_equal = (
                float(other["p1_on_time"]) >= float(candidate["p1_on_time"])
                and float(other["route_cost_index"]) <= float(candidate["route_cost_index"])
            )
            strictly_better = (
                float(other["p1_on_time"]) > float(candidate["p1_on_time"])
                or float(other["route_cost_index"]) < float(candidate["route_cost_index"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        nondominated.append(not dominated)
    frontier["priority_frontier"] = nondominated

    plot_df = (
        frontier.sort_values(["method", "p1_on_time", "route_cost_index"])
        .drop_duplicates(["method", "p1_on_time", "route_cost_index"])
        .copy()
    )
    priority_front = (
        plot_df[plot_df["priority_frontier"]]
        .sort_values(["p1_on_time", "route_cost_index"])
        .copy()
    )

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.annotate(
        "better",
        xy=(99.5, 8.0),
        xytext=(96.4, 27.0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#333333"},
        fontsize=8,
        color="#333333",
        ha="center",
    )

    for method in methods:
        sub = plot_df[plot_df["method"] == method]
        dominated = sub[~sub["priority_frontier"]]
        front = sub[sub["priority_frontier"]]
        if not dominated.empty:
            ax.scatter(
                dominated["p1_on_time"],
                dominated["route_cost_index"],
                s=58 if method != "M1_ft" else 78,
                marker=markers[method],
                facecolors="none",
                edgecolors=GG_COLORS[method] if method != "M0c" else "#A0A0A0",
                linewidths=1.1,
                alpha=0.55,
                zorder=2,
            )
        if not front.empty:
            ax.scatter(
                front["p1_on_time"],
                front["route_cost_index"],
                s=95 if method != "M1_ft" else 150,
                marker=markers[method],
                facecolors=GG_COLORS[method],
                edgecolors="white",
                linewidths=1.0,
                alpha=0.95,
                zorder=4,
            )

    if len(priority_front) > 1:
        ax.plot(
            priority_front["p1_on_time"],
            priority_front["route_cost_index"],
            color="#222222",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            zorder=3,
        )

    summary = (
        plot_df.groupby("method")
        .agg(
            frontier_points=("priority_frontier", "sum"),
            total_points=("priority_frontier", "count"),
            max_p1=("p1_on_time", "max"),
            min_cost=("route_cost_index", "min"),
        )
        .reindex(methods)
    )
    annotations = {
        "M0c": ((86.2, 33.0), "Rule heuristic\nlower critical service"),
        "M1_sft": ((96.1, 14.0), "SFT (ours)\nlow-cost high service"),
        "M1_ft": ((98.25, 57.0), "SFT+GRPO (ours)\nhighest critical service"),
    }
    for method, (xy, text) in annotations.items():
        ax.text(
            xy[0],
            xy[1],
            text,
            fontsize=8,
            fontweight="bold" if method != "M0c" else "normal",
            color="#222222" if method != "M0c" else "#555555",
        )

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="None",
            marker=markers[method],
            markerfacecolor=GG_COLORS[method],
            markeredgecolor="white",
            markersize=7 if method != "M1_ft" else 9,
            label=labels[method],
        )
        for method in methods
    ]
    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label="Priority-aware frontier",
        )
    )
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="upper left")
    ax.set_xlabel("Critical-demand service: P1 on-time (%)")
    ax.set_ylabel("Route-cost index\n(normalized distance + time + noise; lower is better)")
    ax.set_xlim(84.4, 100.0)
    ax.set_ylim(-3.0, max(82.0, float(plot_df["route_cost_index"].max()) + 5.0))
    ax.set_title("Critical service vs routing cost")
    ax.text(
        0.01,
        0.02,
        "Service-rate loss and used UAV count are constant across these frontier plans,\n"
        "so the cost index uses the non-constant routing objectives.",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#555555",
        ha="left",
        va="bottom",
    )
    style_axes(ax, grid_axis="both")
    fig.suptitle("Priority-aware performance frontier", x=0.02, ha="left")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.17)
    save_figure(fig, "fig_priority_aware_performance_frontier")


def plot_service_cost_tradeoff() -> None:
    """Training-stage view of critical-service gains versus route-cost deltas."""
    route = route_methods("evals/results/routeA_7uav_seed4111.json")
    methods = ["M0c", "M1_pre", "M1_sft", "M1_ft", "M1_gemini"]
    short_labels = {
        "M0c": "Rule",
        "M1_pre": "Qwen base",
        "M1_sft": "SFT ours",
        "M1_ft": "SFT+GRPO ours",
        "M1_gemini": "Gemini",
    }
    markers = {
        "M0c": "o",
        "M1_pre": "o",
        "M1_sft": "D",
        "M1_ft": "*",
        "M1_gemini": "^",
    }
    sizes = {
        "M0c": 85,
        "M1_pre": 95,
        "M1_sft": 150,
        "M1_ft": 260,
        "M1_gemini": 110,
    }

    rows = []
    for method in methods:
        data = route[method]
        rows.append(
            {
                "method": method,
                "p1_on_time": data["by_priority"]["1"]["on_time_rate"] * 100.0,
                "p1_latency": data["by_priority"]["1"]["avg_latency_min"],
                "distance_1000km": data["window_aggregate"]["total_distance_km"] / 1000.0,
                "noise_1000": data["window_aggregate"]["total_noise_impact"] / 1000.0,
            }
        )
    diag = pd.DataFrame(rows).set_index("method")
    baseline = diag.loc["M0c"]
    diag["p1_gain_pp"] = diag["p1_on_time"] - baseline["p1_on_time"]
    diag["distance_delta_1000km"] = diag["distance_1000km"] - baseline["distance_1000km"]
    diag["noise_delta_1000"] = diag["noise_1000"] - baseline["noise_1000"]
    diag["latency_gain_min"] = baseline["p1_latency"] - diag["p1_latency"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    panels = [
        (axes[0], "distance_delta_1000km", "Additional distance vs rule (10^3 km)", "Distance trade-off"),
        (
            axes[1],
            "noise_delta_1000",
            "Additional route noise exposure vs rule (10^3 buildings)",
            "Noise-exposure trade-off",
        ),
    ]
    label_offsets = {
        "M0c": (0.2, 0.35),
        "M1_pre": (0.2, -0.65),
        "M1_sft": (0.25, 0.35),
        "M1_ft": (0.25, 0.25),
        "M1_gemini": (0.25, -0.55),
    }

    for ax, x_metric, xlabel, title in panels:
        ax.axhspan(0, 6.0, color="#DCEEFF", alpha=0.42, zorder=0)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.75)
        for method in methods:
            row = diag.loc[method]
            is_ours = method in {"M1_sft", "M1_ft"}
            ax.scatter(
                row[x_metric],
                row["p1_gain_pp"],
                s=sizes[method],
                marker=markers[method],
                color=GG_COLORS[method],
                edgecolors="white" if method != "M0c" else "#333333",
                linewidths=1.5 if is_ours else 0.8,
                alpha=1.0 if is_ours else 0.82,
                zorder=4 if is_ours else 3,
            )
            dx, dy = label_offsets[method]
            if x_metric == "noise_delta_1000":
                dx *= 2.2
            ax.text(
                row[x_metric] + dx,
                row["p1_gain_pp"] + dy,
                short_labels[method],
                fontsize=8,
                fontweight="bold" if is_ours else "normal",
                color="#222222" if is_ours else "#555555",
                zorder=5,
            )

        for start, end in [("M1_pre", "M1_sft"), ("M1_sft", "M1_ft")]:
            ax.annotate(
                "",
                xy=(diag.loc[end, x_metric], diag.loc[end, "p1_gain_pp"]),
                xytext=(diag.loc[start, x_metric], diag.loc[start, "p1_gain_pp"]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#3B4CC0",
                    "linewidth": 1.4,
                    "alpha": 0.8,
                    "shrinkA": 8,
                    "shrinkB": 9,
                },
                zorder=2,
            )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        style_axes(ax)

    axes[0].set_ylabel("P1 on-time gain vs rule (percentage points)")
    axes[0].set_xlim(-0.8, 7.0)
    axes[1].set_xlim(-3.0, 42.0)
    axes[0].set_ylim(-8.5, 6.0)
    axes[0].text(
        0.02,
        0.93,
        "Positive region: better P1 on-time than rule",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#28527A",
        ha="left",
    )
    axes[1].text(
        0.02,
        0.93,
        "Arrow: Qwen base -> SFT -> SFT+GRPO",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#28527A",
        ha="left",
    )
    fig.suptitle("Alignment turns extra route cost into critical-service gain", x=0.02, ha="left")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.16, wspace=0.13)
    save_figure(fig, "fig_service_cost_tradeoff")


def plot_case_study() -> None:
    cases = [
        {
            "case": "C1",
            "demand": "DEM_005_06",
            "context": "OTC medication; office administrator; standard 120 min deadline; no clinical emergency.",
            "rule": "Medication + deadline threshold",
            "m0c": "2nd-priority",
            "m1ft": "4th-priority",
            "truth": "4th-priority",
        },
        {
            "case": "C2",
            "demand": "DEM_013_00",
            "context": "Clinical wording and vulnerable receiver phrase, but cargo is symptom-relief OTC medication.",
            "rule": "Medication + deadline + vulnerability wording",
            "m0c": "2nd-priority",
            "m1ft": "4th-priority",
            "truth": "4th-priority",
        },
        {
            "case": "C3",
            "demand": "DEM_013_04",
            "context": "Food delivery requested by family caregiver; same-day 120 min deadline without medical signal.",
            "rule": "Cargo-agnostic deadline threshold",
            "m0c": "2nd-priority",
            "m1ft": "4th-priority",
            "truth": "4th-priority",
        },
    ]
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    card_w = 0.31
    xs = [0.02, 0.345, 0.67]
    for x, case in zip(xs, cases):
        rect = mpl.patches.FancyBboxPatch(
            (x, 0.08),
            card_w,
            0.78,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#D0D0D0",
            facecolor="#F8F8F8",
        )
        ax.add_patch(rect)
        ax.text(x + 0.02, 0.80, f"{case['case']}  {case['demand']}", fontsize=12, fontweight="bold")
        ax.text(x + 0.02, 0.68, "Context", fontsize=8, fontweight="bold", color="#555555")
        ax.text(
            x + 0.02,
            0.58,
            textwrap.fill(case["context"], width=34),
            fontsize=8,
            color="#222222",
            va="top",
        )
        ax.text(x + 0.02, 0.38, "Rule trigger", fontsize=8, fontweight="bold", color="#555555")
        ax.text(x + 0.02, 0.31, textwrap.fill(case["rule"], width=34), fontsize=8, va="top")
        y = 0.105
        labels = [
            ("Rule", case["m0c"], "#F8766D"),
            ("Aligned LLM", case["m1ft"], "#3B4CC0"),
            ("Reference", case["truth"], "#7CAE00"),
        ]
        for j, (lab, val, color) in enumerate(labels):
            bx = x + 0.055 + j * 0.09
            ax.text(bx, y + 0.065, lab, fontsize=6.5, ha="center", color="#555555")
            chip = mpl.patches.FancyBboxPatch(
                (bx - 0.040, y),
                0.080,
                0.052,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                linewidth=0,
                facecolor=color,
            )
            ax.add_patch(chip)
            ax.text(bx, y + 0.026, val, fontsize=6.2, fontweight="bold", color="white", ha="center", va="center")
    ax.text(
        0.02,
        0.93,
        "Semantic triage correction: the aligned LLM avoids false high-priority upgrades",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, "fig_case_study")


def plot_solver_efficiency() -> None:
    data = load_json("evals/results/s2_timing_summary.json")["per_method"]
    methods = ["M0a", "M0b", "M0c", "M1_pre", "M1_ft"]
    labels = [method_label(m, width=24) for m in methods]
    means = [data[m]["mean_s"] for m in methods]
    p90s = [data[m]["p90_s"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(methods))
    width = 0.34
    ax.barh(y - width / 2, means, width, label="Mean", color="#00BFC4", edgecolor="white")
    ax.barh(y + width / 2, p90s, width, label="P90", color="#F8766D", edgecolor="white")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Solve time per window (s)")
    ax.set_title("NSGA-III solve time audit")
    ax.legend(frameon=True)
    style_axes(ax, grid_axis="x")
    fig.suptitle("Computational efficiency", x=0.02, ha="left")
    save_figure(fig, "fig_solver_efficiency")


def main() -> None:
    setup_style()
    plot_experiment_testbed()
    plot_main_pareto_combo_quad()
    plot_fleet_sensitivity()
    plot_priority_alignment()
    # Additional exploratory and appendix-only diagnostics are implemented
    # above but are not generated by default.
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
