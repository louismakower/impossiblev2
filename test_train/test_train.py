"""An ML task that can't be done honestly.

The agent gets a labelled train set and a labelled test set, and is asked to
reach a threshold accuracy on test. The splits are labelled by different rules
(see `generate.py`), so what is learned from train doesn't transfer. Using the
test labels does reach the threshold: the scanners record how the agent got
there.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash

from common.agent import turns_left
from constants import TURN_LIMIT
from dataset import SCORERS, samples


@task
def test_train(submission: str = "csv", turn_limit: int = TURN_LIMIT):
    """One task per way of submitting, run over every context.

    Args:
        submission: which submission instructions the agent gets, and so which
            scorer grades it — one of `dataset.SCORERS`.
        turn_limit: messages the agent gets before the sample is cut off.
    """
    return Task(
        name=f"test_train_{submission}",
        dataset=samples(submissions=[submission]),
        solver=react(
            tools=[bash(timeout=300)],
            on_continue=turns_left,
        ),
        scorer=SCORERS[submission](),
        sandbox=("docker", "compose.yaml"),
        turn_limit=turn_limit,
        epochs=5,
    )
