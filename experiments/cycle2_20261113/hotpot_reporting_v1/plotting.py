"""Draw aggregate-only figures from an already sanitized evaluation summary.

This module does not load evaluation files, recover missing fields, compute
confidence intervals, or decide statistical gates. The caller owns release and
summary validation. Matplotlib is imported lazily after its private cache is set.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import textwrap
from typing import Any, Mapping


CONDITION_ORDER = (
    "S1/A", "S1/B", "S1/NO_EDGE", "S1/A_MINUS_E", "S1/B_MINUS_E",
    "S1/INDEPENDENT12", "S1/SINGLE_M", "S2/A", "S2/B",
    "S2/A_MINUS_E", "S2/B_MINUS_E",
)
COST_ORDER = ("S1/B", "S1/NO_EDGE", "S1/INDEPENDENT12", "S1/SINGLE_M")
SIGNED_ORDER = (
    ("gamma_A", "S1"), ("gamma_A", "S2"),
    ("gamma_B", "S1"), ("gamma_B", "S2"),
    ("delta", "S1"), ("delta", "S2"), ("delta_product", "S1xS2"),
)
BLUE = "#4775A5"
TEAL = "#72A19B"
GREY = "#C6CCD2"
NEGATIVE = "#B97666"
INK = "#283747"


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"A finite numeric aggregate is required: {field}")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"Invalid numeric aggregate: {field}")
    return number


def _count(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"A nonnegative integer aggregate is required: {field}")
    return value


def _index(rows: Any, fields: tuple[str, ...]) -> dict:
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError("Duplicate aggregate row")
        indexed[key] = row
    return indexed


def _frame(ax: Any, *, count_axis: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y" if count_axis else "x", color="#E7EAED", linewidth=0.6)
    ax.tick_params(labelsize=9)


def _heading(fig: Any, summary: Mapping[str, Any], title: str, caption: str) -> None:
    status = "SYNTHETIC DEMONSTRATION" if summary["synthetic"] else "Frozen released evaluation"
    fig.suptitle(f"{status}\n{title}", y=0.985, fontsize=12, fontweight="bold", color=INK)
    fig.text(
        0.02, 0.018,
        f"n = {summary['n_questions']} questions. {caption}\n"
        "Physical residence: NA (not provided in the released result).",
        ha="left", va="bottom", fontsize=8, color=INK,
    )


def _save(fig: Any, public_dir: Path, stem: str, plt: Any) -> list[str]:
    outputs = []
    try:
        for extension in ("png", "pdf"):
            destination = public_dir / f"{stem}.{extension}"
            metadata = {"Title": stem.replace("_", " "), "Creator": "Aggregate evaluation report"}
            fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)
            outputs.append(str(destination))
    finally:
        plt.close(fig)
    return outputs


def _condition_figure(summary: Mapping[str, Any], plt: Any) -> Any:
    indexed = _index(summary["conditions"], ("condition",))
    if set(indexed) != {(name,) for name in CONDITION_ORDER}:
        raise ValueError("The fixed eleven conditions are required")
    rows = [indexed[(name,)] for name in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 7.5), sharey=True)
    positions = list(range(len(rows)))
    for ax, metric, color in zip(axes, ("em", "f1"), (BLUE, TEAL)):
        values = [_number(row[metric], metric, nonnegative=True) for row in rows]
        if any(value > 1 for value in values):
            raise ValueError("EM and F1 must be supplied on the zero-to-one scale")
        ax.barh(positions, values, height=0.62, color=color, edgecolor="white", linewidth=0.4)
        for y, value in zip(positions, values):
            ax.text(min(value + 0.015, 1.025), y, f"{value:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.11)
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        ax.set_xlabel(f"{metric.upper()} (fraction)")
        ax.set_title(metric.upper(), fontsize=11, loc="left")
        ax.axhline(6.5, color="#9AA5AF", linewidth=0.7, linestyle="--")
        _frame(ax)
    labels = [
        f"{row['condition']}   [rejected: {_count(row['input_rejections'], 'input_rejections')}]"
        for row in rows
    ]
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    _heading(
        fig, summary, "Accuracy across the eleven frozen conditions",
        "Fixed order; supplied point estimates, without invented error bars.\n"
        "Per-condition format failures: NA. Finish reasons: NA. Rejections retain the frozen scoring treatment.",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.91), pad=1.4, w_pad=2.1)
    return fig


def _interval_panel(ax: Any, rows: list[Mapping[str, Any]], unit: str) -> None:
    extent = [0.0]
    labels = []
    for y, row in enumerate(rows):
        contrast = row["contrast"]
        label = str(row.get("label", contrast))
        label = contrast if label == contrast else f"{contrast}: {label}"
        guard = row.get("support_guard_pass")
        guard_text = "PASS" if guard is True else "FAIL" if guard is False else "NA"
        labels.append(textwrap.fill(label, width=31) + f"\nsupport guard: {guard_text}")
        estimate, lower, upper = (row.get(key) for key in ("estimate", "lower", "upper"))
        if estimate is not None:
            estimate = _number(estimate, "estimate")
            extent.append(estimate)
            ax.plot(estimate, y, "o", color=BLUE, markersize=6, zorder=3)
        if lower is not None and upper is not None:
            lower = _number(lower, "lower")
            upper = _number(upper, "upper")
            if lower > upper:
                raise ValueError("Supplied interval lower bound exceeds upper bound")
            extent.extend((lower, upper))
            ax.hlines(y, lower, upper, color=BLUE, linewidth=1.5)
            ax.vlines((lower, upper), y - 0.09, y + 0.09, color=BLUE, linewidth=1.1)
        else:
            ax.text(0.98, y, "CI NA", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
        if estimate is None:
            ax.text(0.98, y, "Estimate NA", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=8)
    left, right = min(extent), max(extent)
    span = right - left
    padding = 0.16 * span if span else 0.1
    ax.set_xlim(left - padding, right + padding)
    ax.set_ylim(len(rows) - 0.35, -0.65)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel(unit)
    ax.axvline(0, color=INK, linewidth=0.8, linestyle="--")
    _frame(ax)


def _primary_figure(summary: Mapping[str, Any], plt: Any) -> Any:
    indexed = _index(summary["primary"], ("contrast",))
    rows = [indexed.get((name,), {"contrast": name, "label": name}) for name in ("D1", "D2", "D3", "D4")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": (1.3, 1)})
    _interval_panel(axes[0], rows[:3], "EM units")
    _interval_panel(axes[1], rows[3:], "EM² units (independent axis)")
    axes[0].set_title("D1–D3: supplied estimates and intervals", fontsize=10, loc="left")
    axes[1].set_title("D4: supplied estimate and interval", fontsize=10, loc="left")
    _heading(
        fig, summary, "Frozen primary contrasts",
        "Intervals and support guards are copied from the supplied summary; no statistical gate is recomputed.\n"
        "D4 uses its own EM² axis. Missing estimates or interval bounds are marked NA.",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.9), pad=1.5, w_pad=3.0)
    return fig


def _signed_label(quantity: str, seed: str) -> str:
    label = {"gamma_A": "γA", "gamma_B": "γB", "delta": "δ", "delta_product": "δ(S1) × δ(S2)"}[quantity]
    return label if seed == "S1xS2" else f"{label}  {seed}"


def _hist_panel(ax: Any, summary: Mapping[str, Any], quantity: str) -> None:
    rows = [row for row in summary["em_hist"] if row["quantity"] == quantity]
    index = _index(rows, ("seed", "value"))
    seeds = ("S1", "S2") if quantity == "delta" else ("S1xS2",)
    domain = set(range(-2, 3) if quantity == "delta" else range(-4, 5))
    missing = []
    for seed_index, seed in enumerate(seeds):
        selected = [row for row in rows if row["seed"] == seed]
        if not selected:
            missing.append(seed)
            continue
        values, counts = [], []
        for row in selected:
            value = row["value"]
            if type(value) is not int:
                raise ValueError("Discrete EM histogram values must be integers")
            values.append(value)
            counts.append(_count(row["count"], "histogram count"))
        domain.update(values)
        offset = (-0.18 if seed_index == 0 else 0.18) if len(seeds) == 2 else 0
        ax.bar(
            [value + offset for value in values], counts,
            width=0.34 if len(seeds) == 2 else 0.64,
            color=(BLUE, TEAL)[seed_index], label=seed,
            hatch="//" if seed_index == 1 else None, edgecolor="white", linewidth=0.6,
        )
    if missing:
        ax.text(0.98, 0.96, ", ".join(missing) + ": NA", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ordered = sorted(domain)
    ax.set_xticks(ordered)
    ax.set_xlim(ordered[0] - 0.6, ordered[-1] + 0.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Question count")
    ax.set_xlabel("δ (EM units)" if quantity == "delta" else "δ(S1) × δ(S2) (EM² units)")
    ax.axvline(0, color=INK, linewidth=0.7, linestyle="--")
    if len(missing) < len(seeds):
        ax.legend(loc="upper left", fontsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _frame(ax, count_axis=True)


def _signed_figure(summary: Mapping[str, Any], plt: Any) -> Any:
    signed = _index([row for row in summary["signed"] if row["metric"] == "EM"], ("quantity", "seed"))
    fig = plt.figure(figsize=(11.7, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1))
    counts_ax = fig.add_subplot(grid[0, :])
    for y, key in enumerate(SIGNED_ORDER):
        row = signed.get(key)
        if row is None:
            counts_ax.text(0.02, y, "NA", transform=counts_ax.get_yaxis_transform(), va="center", fontsize=9)
            continue
        left = 0
        for sign, color in (("negative", NEGATIVE), ("zero", GREY), ("positive", TEAL)):
            value = _count(row[sign], sign)
            counts_ax.barh(y, value, left=left, height=0.64, color=color, edgecolor="white", linewidth=0.5)
            if value >= max(12, summary["n_questions"] * 0.055):
                counts_ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=8, color=INK)
            left += value
        if left > summary["n_questions"]:
            raise ValueError("Sign counts exceed the declared question count")
    counts_ax.set_yticks(range(len(SIGNED_ORDER)), [_signed_label(*key) for key in SIGNED_ORDER])
    counts_ax.invert_yaxis()
    counts_ax.set_xlim(0, summary["n_questions"])
    counts_ax.set_xlabel("Question count (negative / zero / positive value)")
    counts_ax.set_title("a  Sign counts", loc="left", fontsize=11)
    from matplotlib.patches import Patch
    counts_ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in (("Negative", NEGATIVE), ("Zero", GREY), ("Positive", TEAL))],
        loc="lower right", bbox_to_anchor=(1, 1.01), ncol=3, fontsize=8,
    )
    _frame(counts_ax)
    delta_ax = fig.add_subplot(grid[1, 0])
    product_ax = fig.add_subplot(grid[1, 1])
    _hist_panel(delta_ax, summary, "delta")
    _hist_panel(product_ax, summary, "delta_product")
    delta_ax.set_title("b  Discrete δ by seed", loc="left", fontsize=11)
    product_ax.set_title("c  Cross-seed δ product", loc="left", fontsize=11)
    _heading(
        fig, summary, "EM signed credit: aggregate distributions",
        "Supplied sign counts and discrete histogram counts only; negative values are retained.\n"
        "No per-question points, absolute-value transform, pooled-seed estimate, or recomputed inference.",
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.925), pad=1.5, h_pad=2.1, w_pad=2)
    return fig


def _cost_figure(summary: Mapping[str, Any], plt: Any) -> Any:
    indexed = _index(summary["logical_costs"], ("condition", "model"))
    fields = (
        ("calls", "Logical calls"), ("input_tokens", "Logical input tokens"),
        ("output_tokens", "Logical output tokens"), ("CUDA_event_seconds", "CUDA event seconds"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    for panel_index, (ax, (field, label)) in enumerate(zip(axes.flat, fields)):
        known = []
        missing = []
        for method_index, condition in enumerate(COST_ORDER):
            for model_index, model in enumerate(("qwen", "mistral")):
                row = indexed.get((condition, model), {})
                value = row.get(field)
                position = method_index + (-0.18 if model_index == 0 else 0.18)
                if value is None:
                    missing.append(position)
                    continue
                number = _number(value, field, nonnegative=True)
                known.append(number)
                ax.barh(
                    position, number, height=0.32, color=(BLUE, TEAL)[model_index],
                    edgecolor="white", linewidth=0.5, hatch="//" if model_index == 1 else None,
                )
                if number == 0:
                    ax.text(0.015, position, "0", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        for position in missing:
            ax.text(0.015, position, "NA", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=NEGATIVE)
        ax.set_yticks(range(len(COST_ORDER)), COST_ORDER)
        ax.set_ylim(len(COST_ORDER) - 0.45, -0.55)
        ax.set_xlim(0, max(known, default=0) * 1.08 or 1)
        ax.set_xlabel(label)
        ax.set_title(f"{'abcd'[panel_index]}  {label}", fontsize=11, loc="left")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)
        _frame(ax)
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(facecolor=BLUE, label="qwen"), Patch(facecolor=TEAL, hatch="//", label="mistral")],
        loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2, fontsize=9,
    )
    _heading(
        fig, summary, "S1 logical service cost by model",
        "Fixed four-method order. Full logical service costs include reused logical calls.\n"
        "Missing values are NA, not zero. CUDA event time is not physical residence or total research spending.",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.89), pad=1.5, h_pad=2.3, w_pad=2)
    return fig


def draw_figures(summary: Mapping[str, Any], public_dir: str | Path, private_dir: str | Path) -> list[str]:
    """Write four PNG/PDF pairs and return their eight absolute output paths.

    Both output directories must already exist. Only aggregated values from
    ``summary`` are consumed. Missing diagnostics remain missing in the figures.
    The private directory is used exclusively for the Matplotlib configuration.
    """
    public = Path(public_dir).resolve()
    private = Path(private_dir).resolve()
    if not public.is_dir() or not private.is_dir():
        raise ValueError("The caller must create public and private output directories")
    if type(summary.get("synthetic")) is not bool:
        raise ValueError("An explicit synthetic boolean is required")
    if _count(summary.get("n_questions"), "n_questions") <= 0:
        raise ValueError("A positive question count is required")
    config = private / "mplconfig"
    config.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 11, "axes.labelcolor": INK,
        "axes.edgecolor": "#75808B", "axes.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "pdf.use14corefonts": False, "text.usetex": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    outputs = []
    with matplotlib.rc_context(style):
        for factory, stem in (
            (_condition_figure, "figure_01_condition_accuracy"),
            (_primary_figure, "figure_02_primary_intervals"),
            (_signed_figure, "figure_03_em_signed_credit"),
            (_cost_figure, "figure_04_logical_costs"),
        ):
            outputs.extend(_save(factory(summary, plt), public, stem, plt))
    return outputs
