"""Poet x judge grid of mean scores, one per judge variant, plus each judge's
self-preference: the mean it gives its own poems minus the mean it gives
everyone else's.

    uv run poems/analyse.py logs/poems/*.eval

Prints the tables and writes heatmaps to `plots/poems/`.
"""

import math
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from inspect_ai.log import read_eval_log

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import MODELS, SCORE_MAX, SCORE_MIN

PLOTS = Path("plots/poems")
SHORT = {model: name for name, model in MODELS.items()}
ORDER = list(MODELS)


def rows(paths: list[str]) -> pd.DataFrame:
    out = []
    for path in paths:
        log = read_eval_log(path)
        poet = SHORT.get(str(log.eval.model), str(log.eval.model))
        for sample in log.samples or []:
            for score in (sample.scores or {}).values():
                meta = score.metadata or {}
                out.append({
                    "poet": poet,
                    "judge": meta.get("judge"),
                    "told_author": meta.get("told_author"),
                    "told_gt": meta.get("told_gt"),
                    "topic": sample.metadata.get("topic"),
                    "epoch": sample.epoch,
                    # unscored (parse failure) is NaN and drops out of means
                    "score": score.value if isinstance(score.value, (int, float)) else math.nan,
                })
    return pd.DataFrame(out)


def variant_name(told_author: bool, told_gt: bool) -> str:
    return f"{'author' if told_author else 'blind'}{'-gt' if told_gt else ''}"


def grid(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: judge, columns: poet, values: mean score."""
    table = df.pivot_table(index="judge", columns="poet", values="score", aggfunc="mean")
    order = [m for m in ORDER if m in table.index] + sorted(set(table.index) - set(ORDER))
    cols = [m for m in ORDER if m in table.columns] + sorted(set(table.columns) - set(ORDER))
    return table.reindex(index=order, columns=cols)


def self_preference(table: pd.DataFrame) -> pd.Series:
    """Per judge: score for own poems minus mean score for the other poets'."""
    return pd.Series({
        judge: table.loc[judge, judge] - table.loc[judge].drop(judge).mean()
        for judge in table.index
        if judge in table.columns
    })


def heatmap(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    # one hue, light to dark: the cells encode magnitude only
    image = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", vmin=SCORE_MIN, vmax=SCORE_MAX)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("poet")
    ax.set_ylabel("judge")
    ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for i, judge in enumerate(table.index):
        for j, poet in enumerate(table.columns):
            value = table.loc[judge, poet]
            if pd.isna(value):
                continue
            dark = (value - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) > 0.6
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if dark else "#222",
                    fontweight="bold" if judge == poet else None)
    fig.colorbar(image, ax=ax, shrink=0.8, label="mean score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(paths: list[str]) -> None:
    df = rows(paths)
    if df.empty:
        sys.exit("no scores found")
    PLOTS.mkdir(parents=True, exist_ok=True)
    unscored = df["score"].isna().sum()
    print(f"{len(df)} scores, {unscored} unscored\n")
    for told_author in (False, True):
        for told_gt in (False, True):
            name = variant_name(told_author, told_gt)
            subset = df[(df["told_author"] == told_author) & (df["told_gt"] == told_gt)]
            if subset.empty:
                continue
            table = grid(subset)
            print(f"== {name} ==  (rows: judge, columns: poet)")
            print(table.round(2).to_string())
            print("self-preference:", self_preference(table).round(2).to_dict(), "\n")
            heatmap(table, f"mean score, {name}", PLOTS / f"grid_{name}.png")
    print(f"heatmaps in {PLOTS}/")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in sorted(Path("logs/poems").glob("*.eval"))])
