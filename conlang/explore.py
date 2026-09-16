"""Interactive scanner explorer for the conlang grid, see `common.explore`.

Usage:
    uv run conlang/explore.py logs/conlang/*.eval
    uv run conlang/explore.py logs/conlang/*.eval --out plots/conlang.html
"""

from constants import REPORT_PATH
from dataset import CONTEXTS, GRADING_KEYS

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """The report, which is all the judges see, and the benchmark result."""
    blocks = report_block(sample, REPORT_PATH)
    benchmark = sample.scores.get("benchmark_scorer") if sample.scores else None
    if benchmark:
        blocks.append({"title": "benchmark", "text": benchmark.explanation or ""})
    return blocks


if __name__ == "__main__":
    main(GRADING_KEYS, CONTEXTS, evidence, out="plots/conlang.html")
