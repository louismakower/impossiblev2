"""Interactive scanner explorer for the test_train grid, see `common.explore`.

Usage:
    uv run test_train/explore.py logs/test_train/*.eval
    uv run test_train/explore.py logs/test_train/*.eval --out plots/test_train.html
"""

from dataset import CONTEXTS, GRADINGS
from variant import BASELINE, VARIANTS

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """The report. The predictions file has nothing to show beyond its
    accuracy, which the predictions scorer's explanation on the card carries.
    Where the report lives depends on the sample's variant; logs from before
    the variant axis are the baseline."""
    variant = VARIANTS[(sample.metadata or {}).get("variant", BASELINE.name)]
    return report_block(sample, variant.report_path)


if __name__ == "__main__":
    main(GRADINGS, CONTEXTS, evidence, out="plots/test_train.html")
