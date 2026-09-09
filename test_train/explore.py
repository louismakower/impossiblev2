"""Interactive scanner explorer for the test_train grid, see `common.explore`.

Usage:
    uv run test_train/explore.py logs/test_train/*.eval
    uv run test_train/explore.py logs/test_train/*.eval --out plots/test_train.html
"""

from constants import REPORT_PATH
from dataset import CONTEXTS, GRADINGS

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """The report. The predictions file has nothing to show beyond its
    accuracy, which the predictions scorer's explanation on the card carries."""
    return report_block(sample, REPORT_PATH)


if __name__ == "__main__":
    main(GRADINGS, list(CONTEXTS), evidence, out="plots/test_train.html")
