"""Interactive scanner explorer for the tinker_finetune grid, see `common.explore`.

Usage:
    uv run tinker_finetune/explore.py logs/tinker_finetune/*.eval
    uv run tinker_finetune/explore.py logs/tinker_finetune/*.eval --out plots/tinker_finetune.html
"""

from constants import REPORT_PATH
from dataset import CONTEXTS, GRADINGS

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """The report, which is all the judges see."""
    return report_block(sample, REPORT_PATH)


if __name__ == "__main__":
    main(GRADINGS, CONTEXTS, evidence, out="plots/tinker_finetune.html")
