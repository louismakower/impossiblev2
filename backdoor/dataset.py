"""The grid of samples: each context crossed with what the agent is told about grading.

A sample's prompt is three parts joined together — the scenario (how eval-ey
it feels, from `context.py`, including how to submit: the same everywhere, the
conversation JSON and a report), what the agent is told about how those are
graded, and the turn limit. Every sample is graded by the same scorers whatever
it was told, so the whole grid fits in one task: see `backdoor_hosted.py`.
"""

from inspect_ai.dataset import Sample

from constants import TARGET, TURN_LIMIT
from context import (
    HARD_AGENT_PROMPT,
    HARD_GRADING,
    SOFT_AGENT_PROMPT,
    SOFT_GRADING,
)
from common.agent import SOFT_TURN_LIMIT, HARD_TURN_LIMIT

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
    return " ".join(
        part for part in (scenario, grading_prompt[grading], turn_limit)
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
            target=TARGET,
            metadata={"context": context, "grading": grading},
        )
        for context in contexts or CONTEXTS
        for grading in gradings or GRADINGS
    ]


if __name__ == "__main__":
    for context in CONTEXTS:
        for grading in GRADINGS:
            print("\n\n" + "*"*10 + context.upper() + " " + grading.upper() + "*"*10 + "\n\n")
            print(agent_prompt(context, grading))
            input()