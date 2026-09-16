"""Interactive scanner explorer for the shared_file grid, see `common.explore`.

Usage:
    uv run shared_file/explore.py logs/shared_file/*.eval
    uv run shared_file/explore.py logs/shared_file/*.eval --out plots/shared_file.html
"""

from constants import LINES_PATH
from dataset import CONTEXTS, GRADING_KEYS

from common.explore import main


def evidence(sample) -> list[dict]:
    """The file both agents left, and the line scorer's reading of it."""
    lines = sample.scores.get("lines_scorer") if sample.scores else None
    if not lines:
        return []
    return [
        {"title": "lines", "text": lines.explanation or ""},
        {"title": f"shared file ({LINES_PATH})", "text": lines.answer or ""},
    ]


if __name__ == "__main__":
    main(GRADING_KEYS, CONTEXTS, evidence, out="plots/shared_file.html")
