"""The reward function as agent_a sees it: its own reward and the total
against the lines it writes, one panel per bonus factor.

    uv run shared_file/reward_plot.py

Assumes the file ends up full, so agent_b has the remaining lines. Writes
`plots/shared_file/reward.png`. No logs involved.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import BONUS_FACTORS
from reward import AGENTS, BONUS_RANGE, MAX_LINES, rewards

PLOTS = Path("plots/shared_file")
A, B = AGENTS
# categorical slots 1 and 2 of the reference palette
COLOUR_A = "#2a78d6"
COLOUR_TOTAL = "#eb6834"


def curves(bonus_factor: float) -> tuple[list[int], list[float], list[float]]:
    lines = list(range(MAX_LINES + 1))
    own, total = [], []
    for a in lines:
        reward = rewards({A: a, B: MAX_LINES - a}, bonus_factor)
        own.append(reward[A])
        total.append(sum(reward.values()))
    return lines, own, total


def figure(path: Path) -> None:
    fig, axes = plt.subplots(1, len(BONUS_FACTORS), figsize=(3.4 * len(BONUS_FACTORS) + 0.6, 3.4),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(top=0.78, wspace=0.12)
    for ax, bonus in zip(axes[0], BONUS_FACTORS):
        lines, own, total = curves(bonus)
        ax.axvspan(*BONUS_RANGE, color="#e6e5e1", zorder=0)
        ax.plot(lines, total, color=COLOUR_TOTAL, linewidth=2, label="total (both agents)")
        ax.plot(lines, own, color=COLOUR_A, linewidth=2, label=A)
        ax.set_title(f"bonus x{bonus:g}", loc="left")
        ax.set_xlabel(f"lines written by {A}")
        ax.set_xlim(0, MAX_LINES)
        ax.set_xticks([0, BONUS_RANGE[0], BONUS_RANGE[1], MAX_LINES])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("reward")
    axes[0][0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"reward against {A}'s line count, {B} writing the rest "
                 f"(shaded: bonus range {BONUS_RANGE[0]}-{BONUS_RANGE[1]})", x=0.02, y=0.95, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / "reward.png"
    figure(path)
    print(f"figure: {path}")
