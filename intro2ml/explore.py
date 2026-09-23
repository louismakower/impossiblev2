"""Interactive scanner explorer for the intro2ml grid, see `common.explore`.

Usage:
    uv run intro2ml/explore.py logs/intro2ml/*.eval --scans logs/intro2ml/scans
    uv run intro2ml/explore.py logs/intro2ml/*.eval --out plots/intro2ml.html
"""

import json

from dataset import CONTEXTS, GRADINGS
from targets import TARGETS
from variant import BASELINE, VARIANTS

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """Both graded artifacts: the accuracies in the results file, next to the
    example report's, and the report the judges read."""
    variant = VARIANTS[(sample.metadata or {}).get("variant", BASELINE.name)]
    return results_block(sample, variant.results_path) + report_block(sample, variant.report_path)


def results_block(sample, path: str) -> list[dict]:
    """The results scorer keeps the accuracies it read as its score's answer;
    its explanation, which the card shows anyway, says why the file failed."""
    score = sample.scores.get("results_scorer") if sample.scores else None
    if not score or not score.answer:
        return [{"title": "no results", "text": f"The agent never wrote a gradable results file at {path}."}]
    accuracies = json.loads(score.answer)
    lines = [
        f"{dataset}_{mode:<10} {accuracies[f'{dataset}_{mode}']:.4f}   (example report {target:.3f})"
        for dataset, modes in TARGETS.items()
        for mode, target in modes.items()
    ]
    return [{"title": f"results ({path})", "text": "\n".join(lines)}]


if __name__ == "__main__":
    main(GRADINGS, CONTEXTS, evidence, out="plots/intro2ml.html")
