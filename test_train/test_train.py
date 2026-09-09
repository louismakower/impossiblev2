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
from dataset import samples
from scorer import predictions_scorer, report_judge, report_trajectory_judge


@task
def test_train(turn_limit: int = TURN_LIMIT):
    """The whole grid, every sample graded by the same three scorers: the
    predictions file the agent writes, an LLM judge reading its report, and
    an LLM judge reading the report alongside the trajectory.

    Args:
        turn_limit: messages the agent gets before the sample is cut off.
    """
    return Task(
        dataset=samples(),
        solver=react(
            tools=[bash(timeout=300)],
            on_continue=turns_left,
        ),
        scorer=[predictions_scorer(), report_judge(), report_trajectory_judge()],
        sandbox=("docker", "compose.yaml"),
        turn_limit=turn_limit,
        epochs=5,
    )
