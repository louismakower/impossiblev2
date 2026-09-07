"""Interactive scanner explorer for the test_train grid, see `common.explore`.

Usage:
    uv run test_train/explore.py logs/test_train-grid/*.eval
    uv run test_train/explore.py logs/test_train-grid/*.eval --out plots/test_train.html
"""

from constants import REPORT_PATH
from dataset import CONTEXTS, SCORERS

from common.explore import main, report_block


def evidence(sample, submission: str, score) -> list[dict]:
    """The report, for the submissions that write one.

    A csv submission has nothing to show beyond the scorer's accuracy, which
    the sample's card carries anyway.
    """
    if submission == "csv":
        return []
    return report_block(score, REPORT_PATH)


if __name__ == "__main__":
    main(list(SCORERS), list(CONTEXTS), evidence, out="plots/test_train.html")
