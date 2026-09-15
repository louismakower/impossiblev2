"""One cell for now: the single prompt in `context.py`, with the shared turn
limit sentence appended.

The data (a Pile of Law slice and LegalBench, built by `prepare_data.py`) is
mounted into the sandbox by `compose.yaml`, so the sample carries no files.
The sample's metadata names the grid cell the way the other tasks do, so the
analysis code can group by it once more cells exist; contexts, gradings and
variants can be crossed here later the way `test_train/dataset.py` does.
"""

from inspect_ai.dataset import Sample

from common.agent import HARD_TURN_LIMIT
from constants import TURN_LIMIT
from context import agent_prompt

# the one cell so far
VARIANT = "baseline"
CONTEXTS = ["hard"]
GRADINGS = ["none"]


def samples() -> list[Sample]:
    context, grading = CONTEXTS[0], GRADINGS[0]
    return [
        Sample(
            id=f"{context}-{grading}",
            input="\n\n".join(
                [agent_prompt().strip(), HARD_TURN_LIMIT.format(turn_limit=TURN_LIMIT)]
            ),
            metadata={"variant": VARIANT, "context": context, "grading": grading},
        )
    ]
