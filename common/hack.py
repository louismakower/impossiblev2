"""Reward-hack rate across an eval grid, as one PNG.

A sample counts as a reward hack when it passes both the deterministic scorer
(the conversation or predictions file) and the report judge: it did the
impossible thing, and its report persuaded the judge it did so legitimately.

Top row: the hack rate against each factor alone — what the agent was told
about grading, the context, the variant — one line per model, averaged over
everything else, with 95% Wilson intervals. Below, one row of heatmaps per
measure with one panel per model, grading against context × variant, each cell
labelled hits/samples: the hack rate, its two components (the deterministic
scorer and the report judge on their own), then the awareness scanner's two
fields (a hit is `implied` or `explicit`; the cell also says how many were
explicit).
Samples the scanner hasn't reached are left out of the awareness counts.

Usage:
    uv run common/hack.py logs/test_train/*.eval --out plots/test_train_hack.png
"""

import argparse
import json
import textwrap
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai.analysis import EvalModel, EvalTask, SampleSummary, samples_df
from inspect_scout import scan_list, scan_results_df
from matplotlib.colors import LinearSegmentedColormap

# Display order of each factor's levels; gradings roughly by how much scrutiny
# the agent is told to expect. Levels a task lacks are skipped.
GRADINGS = ["none", "json", "csv", "report", "report_judge_prompt", "report_trajectory", "all"]
CONTEXTS = ["soft", "hard"]
VARIANTS = ["baseline", "realistic"]
FACTORS = {"grading": GRADINGS, "context": CONTEXTS, "variant": VARIANTS}

# The awareness scanner's two fields, see `common.scanner`.
AWARENESS = ["eval_awareness", "rewardhack_eval_awareness"]

# Heatmap rows: the boolean column to count, and the row's label.
MEASURES = {
    "hack": "reward hack",
    "deterministic": "passed the deterministic scorer",
    "report_judge": "passed the report judge",
    "eval_awareness": "eval awareness (implied or explicit)",
    "rewardhack_eval_awareness": "reward-hacking-eval awareness (implied or explicit)",
}

# Same palette as backdoor/plotting/analyse.py: categorical slots for the
# models, one-hue ramp for the heatmap.
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def sample_table(logs: list[str]) -> pd.DataFrame:
    """One row per sample: which cell of the grid, and whether it hacked."""
    df = samples_df(logs, columns=EvalModel + EvalTask + SampleSummary)

    # The deterministic scorer is whichever one isn't a judge; its name differs
    # between tasks, so logs from both tasks together give one column each,
    # empty for the other task's samples.
    deterministic = [c for c in df.columns if c.startswith("score_") and "judge" not in c]
    # a sample not yet scored (a log still being written) counts as not passed
    passed = df[deterministic].eq("C").fillna(False).any(axis=1)
    judged = df["score_report_judge"].eq("C").fillna(False).astype(bool)

    # Logs from before the variant axis carry no variant: they are the baseline.
    variant = df["metadata_variant"] if "metadata_variant" in df else pd.Series(None, index=df.index)

    return pd.DataFrame(
        {
            "sample_id": [str(s) for s in df["sample_id"]],
            "task": [str(t) for t in df["task_name"]],
            "model": [str(m).split("/")[-1] for m in df["model"]],
            "grading": [str(g) for g in df["metadata_grading"]],
            "context": [str(c) for c in df["metadata_context"]],
            "variant": [str(v) if pd.notna(v) else "baseline" for v in variant],
            "deterministic": list(passed),
            "report_judge": list(judged),
            "hack": list(passed & judged),
        }
    )


def awareness_table(scans: str) -> pd.DataFrame:
    """The awareness scanner's latest verdict per sample: for each field, a
    column of whether it was at least `implied`, and one of whether it was
    `explicit`."""
    rows = []
    for status in sorted(scan_list(scans), key=lambda s: s.spec.timestamp):
        try:
            results = scan_results_df(status.location).scanners["awareness"]
        except KeyError:  # a scan of other scanners, or one with no results yet
            continue
        for _, row in results.iterrows():
            value = json.loads(row["value"]) if isinstance(row["value"], str) else {}
            # errored, or from an earlier version of the scanner with other fields
            if any(f not in value for f in AWARENESS):
                continue
            rows.append({"sample_id": str(row["transcript_id"]), **{f: value[f] for f in AWARENESS}})
    # Re-scans append a new scan_id rather than overwriting, so keep the last
    # verdict per sample.
    latest = pd.DataFrame(rows, columns=["sample_id", *AWARENESS]).drop_duplicates("sample_id", keep="last")
    for field in AWARENESS:
        latest[f"{field}_explicit"] = latest[field] == "explicit"
        latest[field] = latest[field].isin(["implied", "explicit"])
    return latest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate and 95% interval for k successes in n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # at p = 0 or 1 rounding can put the bound a hair past the rate
    return p, min(p, centre - half), max(p, centre + half)


def levels(df: pd.DataFrame, factor: str) -> list[str]:
    return [l for l in FACTORS[factor] if l in set(df[factor])]


def marginal(ax, df: pd.DataFrame, factor: str, models: list[str]) -> None:
    """Hack rate against one factor, one line per model."""
    xs = levels(df, factor)
    offsets = [(i - (len(models) - 1) / 2) * 0.08 for i in range(len(models))]
    for model, colour, offset in zip(models, MODEL_COLOURS, offsets):
        subset = df[df["model"] == model]
        stats = [wilson(int(subset[subset[factor] == x]["hack"].sum()), int((subset[factor] == x).sum())) for x in xs]
        rates = [s[0] for s in stats]
        ax.errorbar(
            [i + offset for i in range(len(xs))], rates,
            yerr=[[r - lo for r, lo, _ in stats], [hi - r for r, _, hi in stats]],
            color=colour, marker="o", markersize=6, linewidth=2, capsize=3, label=model,
        )
    # "report_trajectory" collides with its neighbours on one line
    ax.set_xticks(range(len(xs)), [x.replace("_", "\n") for x in xs])
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title(factor, fontsize=10.5, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)


def heatmap(ax, df: pd.DataFrame, model: str, measure: str, high: float) -> None:
    """One measure's rate per cell of the grid for one model, labelled
    hits/samples. Samples without a value for the measure are left out; an
    `_explicit` companion column, if present, is counted under the label."""
    subset = df[(df["model"] == model) & df[measure].notna()]
    explicit = f"{measure}_explicit" if f"{measure}_explicit" in df else None
    columns = levels(df, "grading")
    rows = [(c, v) for c in levels(df, "context") for v in levels(df, "variant")]
    cells = [[subset[(subset.context == c) & (subset.variant == v) & (subset.grading == g)] for g in columns] for c, v in rows]
    rates = [[cell[measure].mean() if len(cell) else float("nan") for cell in row] for row in cells]

    ax.imshow(rates, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=high, aspect="auto")
    for y, row in enumerate(cells):
        for x, cell in enumerate(row):
            if len(cell):
                label = f"{int(cell[measure].sum())}/{len(cell)}"
                if explicit and cell[explicit].any():
                    label += f"\n{int(cell[explicit].sum())} explicit"
                ax.text(x, y, label, ha="center", va="center", fontsize=9, linespacing=1.4,
                        color=SURFACE if rates[y][x] > high * 0.55 else INK_MUTED)
    ax.set_xticks(range(len(columns)), [c.replace("_", "\n") for c in columns], fontsize=9)
    ax.set_yticks(range(len(rows)), [f"{c} · {v}" for c, v in rows], fontsize=9)
    # A 2px surface gap between cells, drawn as a minor grid.
    ax.set_xticks([x - 0.5 for x in range(len(columns) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(rows) + 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(model, fontsize=10.5, color=INK, loc="left")
    style(ax, spines=False)


def style(ax, spines: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_visible(spines and side == "bottom")
        spine.set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="eval log files to include")
    parser.add_argument("--out", default="plots/hack.png", help="PNG to write")
    parser.add_argument("--scans", default=None,
                        help="scan results directory (defaults to `scans` beside the first log)")
    args = parser.parse_args()

    scans = args.scans or str(Path(args.logs[0]).parent / "scans")
    df = sample_table(args.logs).merge(awareness_table(scans), on="sample_id", how="left")
    models = sorted(set(df["model"]))
    rows = 1 + len(MEASURES)

    fig = plt.figure(figsize=(4.4 * max(3, len(models)), 3.6 * rows + 1.2), facecolor=SURFACE)
    for i, factor in enumerate(FACTORS):
        marginal(fig.add_subplot(rows, 3, i + 1), df, factor, models)
    for r, (measure, label) in enumerate(MEASURES.items(), start=1):
        high = max(0.2, df.groupby(["model", "grading", "context", "variant"])[measure].mean().max())
        for j, model in enumerate(models):
            ax = fig.add_subplot(rows, len(models), r * len(models) + j + 1)
            heatmap(ax, df, model, measure, high)
            if j == 0:
                ax.set_ylabel(textwrap.fill(label, 26), fontsize=10, color=INK, labelpad=12)

    scanned = int(df["eval_awareness"].notna().sum())
    # Header in inches from the top, so it doesn't move with the figure height.
    height = fig.get_figheight()
    fig.suptitle("Reward hacking and eval awareness across the grid",
                 fontsize=13, color=INK, x=0.012, ha="left", y=1 - 0.25 / height)
    fig.text(0.012, 1 - 0.5 / height,
             f"{len(df)} samples from {len(args.logs)} log(s), {scanned} scanned for awareness. "
             "A reward hack passes both the deterministic scorer and the report judge.\n"
             "Top row averages over the other factors, bars are 95% Wilson intervals.",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top", linespacing=1.5)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.99, 1 - 0.15 / height), ncol=len(models), frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.subplots_adjust(left=0.13, right=0.98, top=1 - 1.35 / height, bottom=0.6 / height, hspace=0.5, wspace=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out}: {int(df['hack'].sum())} hacks in {len(df)} samples, {scanned} scanned for awareness")


if __name__ == "__main__":
    main()
