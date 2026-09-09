"""The grid of samples: each context crossed with what the agent is told about grading.

A sample's prompt is three parts joined together — the scenario (how eval-ey
it feels, from `context.py`, including how to submit: the same everywhere, the
predictions CSV and a report), what the agent is told about how those are
graded, and the turn limit. Every sample is graded by the same scorers whatever
it was told, so the whole grid fits in one task: see `test_train.py`. Every
sample carries the same two data files.
"""

from pathlib import Path

from inspect_ai.dataset import Sample

from common.agent import HARD_TURN_LIMIT, SOFT_TURN_LIMIT
from constants import TEST_PATH, TRAIN_PATH, TURN_LIMIT
from context import (
    HARD_AGENT_PROMPT,
    HARD_GRADING,
    SOFT_AGENT_PROMPT,
    SOFT_GRADING,
)

DATA_DIR = Path(__file__).parent / "data"

# copied into the sandbox before the agent starts
FILES = {
    TRAIN_PATH: str(DATA_DIR / "train.csv"),
    TEST_PATH: str(DATA_DIR / "test.csv"),
}

# context name -> (scenario, grading prompt, turn limit)
CONTEXTS = {
    "soft": (SOFT_AGENT_PROMPT, SOFT_GRADING, SOFT_TURN_LIMIT.format(turn_limit=TURN_LIMIT)),
    "hard": (HARD_AGENT_PROMPT, HARD_GRADING, HARD_TURN_LIMIT.format(turn_limit=TURN_LIMIT)),
}

# what the agent can be told about grading; the soft and hard dicts share these keys
GRADINGS = list(HARD_GRADING)


def agent_prompt(context: str, grading: str) -> str:
    """The full prompt for one cell of the grid."""
    scenario, grading_prompt, turn_limit = CONTEXTS[context]
    return "\n\n".join(
        part.strip() for part in (scenario, grading_prompt[grading], turn_limit)
    )


def samples(
    contexts: list[str] | None = None,
    gradings: list[str] | None = None,
) -> list[Sample]:
    """Samples for every context/grading pair, defaulting to all of them.

    The pair is recorded in each sample's metadata as well as its id, so scores
    can be grouped by either when analysing a log.
    """
    return [
        Sample(
            id=f"{context}-{grading}",
            input=agent_prompt(context, grading),
            files=FILES,
            metadata={"context": context, "grading": grading},
        )
        for context in contexts or CONTEXTS
        for grading in gradings or GRADINGS
    ]
