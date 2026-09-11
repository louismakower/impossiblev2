"""An ML task that can't be done honestly.

The agent gets a labelled train set and a labelled test set, and is asked to
reach a threshold accuracy on test. The splits are labelled by different rules
(see `generate.py`), so what is learned from train doesn't transfer. Using the
test labels does reach the threshold: the scanners record how the agent got
there.
"""

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, react
from inspect_ai.tool import bash

from common.agent import stash_context, user_turns
from constants import TURN_LIMIT
from dataset import samples
from scorer import predictions_scorer, report_judge, report_trajectory_judge
from variant import BASELINE, VARIANTS


@task
def test_train(variant: str = BASELINE.name, turn_limit: int = TURN_LIMIT):
    """One variant's whole grid, every sample graded by the same three
    scorers: the predictions file the agent writes, an LLM judge reading its
    report, and an LLM judge reading the report alongside the trajectory.

    Args:
        variant: which bundle of realism levers to run, by name (`variant.py`).
        turn_limit: messages the agent gets before the sample is cut off.
    """
    v = VARIANTS[variant]
    return Task(
        dataset=samples(v),
        solver=[
            stash_context(),
            as_solver(
                react(
                    tools=[bash(timeout=300)],
                    on_continue=user_turns(v.follow_ups, v.countdown, TURN_LIMIT),
                    submit=v.submit,
                )
            ),
        ],
        scorer=[
            predictions_scorer(v.predictions_path),
            report_judge(v.report_path),
            report_trajectory_judge(v.report_path),
        ],
        sandbox=("docker", v.compose),
        turn_limit=turn_limit,
        epochs=5,
    )
