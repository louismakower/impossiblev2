"""Head start x bonus factor grid of the mean reward gap, agent_a minus
agent_b, one heatmap per model.

    uv run shared_file/analyse.py logs/shared_file/2026-09-18T13-*.eval

Prints one table per model and writes a row of heatmaps to
`plots/shared_file/diff.png`. Cells are averaged over epochs; a cell with no
finished samples yet is left blank.
"""

import sys
from pathlib import Path

import matplotlib
import pandas as pd
from inspect_ai.log import read_eval_log

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import BONUS_FACTORS, HEAD_STARTS
from reward import AGENTS

PLOTS = Path("plots/shared_file")
# short name -> model id, in display order
MODELS = {
    "kimi": "openrouter/moonshotai/kimi-k3",
    "astra": "openrouter/openai/gpt-6-astra",
    "fable": "anthropic/claude-fable-5-1",
    "glm": "openrouter/z-ai/glm-5.3",
}
SHORT = {model: name for name, model in MODELS.items()}
A, B = AGENTS


def rows(paths: list[str]) -> pd.DataFrame:
    out = []
    for path in paths:
        log = read_eval_log(path)
        model = SHORT.get(str(log.eval.model), str(log.eval.model))
        for sample in log.samples or []:
            score = (sample.scores or {}).get("lines_scorer")
            if score is None or not isinstance(score.value, dict):
                continue
            out.append({
                "model": model,
                "head_start": sample.metadata["head_start"],
                "bonus_factor": sample.metadata["bonus_factor"],
                "epoch": sample.epoch,
                A: score.value[A],
                B: score.value[B],
                "diff": score.value[A] - score.value[B],
            })
    return pd.DataFrame(out)


def grid(df: pd.DataFrame, value: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Rows: head start, columns: bonus factor, in the sweep's order."""
    table = df.pivot_table(index="head_start", columns="bonus_factor", values=value, aggfunc=aggfunc)
    return table.reindex(index=HEAD_STARTS, columns=BONUS_FACTORS)


def heatmap(ax, table: pd.DataFrame, title: str, limit: float, ylabel: bool):
    """One head start x bonus panel; returns the image for a shared colourbar."""
    # signed gap: two hues either side of a neutral midpoint at zero
    image = ax.imshow(table.to_numpy(dtype=float), cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(table.columns)), [f"x{b:g}" for b in table.columns])
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("bonus factor")
    if ylabel:
        ax.set_ylabel("head start (lines)")
    ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for i, head in enumerate(table.index):
        for j, bonus in enumerate(table.columns):
            value = table.loc[head, bonus]
            if pd.isna(value):
                continue
            dark = abs(value) / limit > 0.6
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if dark else "#222")
    return image


def figure(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """One panel per model in a row, sharing a colour scale symmetric about zero."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables) + 1.2, 3.8), squeeze=False)
    fig.subplots_adjust(top=0.78, wspace=0.3)
    limit = max(table.abs().max().max() for table in tables.values())
    limit = max(limit, 0.05) if pd.notna(limit) else 0.05
    image = None
    for i, (ax, (model, table)) in enumerate(zip(axes[0], tables.items())):
        image = heatmap(ax, table, model, limit, ylabel=i == 0)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label=f"mean reward, {A} minus {B}")
    fig.suptitle(f"reward gap ({A} minus {B}) by head start and bonus factor", x=0.02, y=0.95, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(paths: list[str]) -> None:
    df = rows(paths)
    if df.empty:
        sys.exit("no lines_scorer scores found")
    PLOTS.mkdir(parents=True, exist_ok=True)
    order = [m for m in MODELS if m in set(df["model"])] + sorted(set(df["model"]) - set(MODELS))
    tables = {}
    for model in order:
        subset = df[df["model"] == model]
        tables[model] = grid(subset, "diff")
        print(f"== {model} ==  {len(subset)} samples  (rows: head start, columns: bonus factor)")
        print("mean diff:")
        print(tables[model].round(3).to_string())
        print("samples per cell:")
        print(grid(subset, "diff", "count").fillna(0).astype(int).to_string(), "\n")
    path = PLOTS / "diff.png"
    figure(tables, path)
    print(f"figure: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
