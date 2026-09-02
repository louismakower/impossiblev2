"""The grid of samples: each context crossed with each way of submitting.

A sample's prompt is three parts joined together — the scenario (how eval-ey
it feels, from `context.py`), how the agent is told to submit, and the turn
limit. The submission key also names the scorer that grades it, so the two
always agree:

    from dataset import SCORERS, samples

    Task(dataset=samples(submissions=["report"]), scorer=SCORERS["report"]())

Inspect scores a whole task with one scorer, so a task should take samples for
a single submission key. Crossing the grid means running one task per key,
e.g. with an eval set.
"""

from inspect_ai.dataset import Sample

from common import TARGET
from context import (
    HARD_AGENT_PROMPT,
    HARD_SUBMISSION_INSTRUCTIONS,
    HARD_TURN_LIMIT,
    SOFT_AGENT_PROMPT,
    SOFT_SUBMISSION_INSTRUCTIONS,
    SOFT_TURN_LIMIT,
)
from scorer import (
    json_scorer,
    llm_report_judge,
    llm_report_trajectory_judge,
    tool_reader,
)

# context name -> (scenario, submission instructions, turn limit)
CONTEXTS = {
    "soft": (SOFT_AGENT_PROMPT, SOFT_SUBMISSION_INSTRUCTIONS, SOFT_TURN_LIMIT),
    "hard": (HARD_AGENT_PROMPT, HARD_SUBMISSION_INSTRUCTIONS, HARD_TURN_LIMIT),
}

# submission key -> the scorer that grades that way of submitting
SCORERS = {
    "json": json_scorer,
    "tool": tool_reader,
    "report": llm_report_judge,
    "report_trajectory": llm_report_trajectory_judge,
}


def agent_prompt(context: str, submission: str) -> str:
    """The full prompt for one cell of the grid."""
    scenario, instructions, turn_limit = CONTEXTS[context]
    return "\n\n".join(
        part.strip() for part in (scenario, instructions[submission], turn_limit)
    )


def samples(
    contexts: list[str] | None = None,
    submissions: list[str] | None = None,
) -> list[Sample]:
    """Samples for every context/submission pair, defaulting to all of them.

    The pair is recorded in each sample's metadata as well as its id, so scores
    can be grouped by either when analysing a log.
    """
    for context in contexts or []:
        if context not in CONTEXTS:
            raise ValueError(
                f"Unknown context {context!r}, expected one of {list(CONTEXTS)}."
            )
    for submission in submissions or []:
        if submission not in SCORERS:
            raise ValueError(
                f"Unknown submission {submission!r}, expected one of {list(SCORERS)}."
            )

    return [
        Sample(
            id=f"{context}-{submission}",
            input=agent_prompt(context, submission),
            target=TARGET,
            metadata={"context": context, "submission": submission},
        )
        for context in contexts or CONTEXTS
        for submission in submissions or SCORERS
    ]
