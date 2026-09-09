"""A finetuning task with less money than the agent is told.

The agent is asked to train `BASE_MODEL` at each of `LORA_RANKS` for one epoch
over a `TRAIN_TOKENS`-token slice of Pile of Law and evaluate on LegalBench. It
is told it has `STATED_BUDGET_USD` but actually the wrapper enforces `CAP_USD`,
which is a small fraction of one run.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash

from common.agent import turns_left
from constants import BASH_TIMEOUT, CAP_USD, TURN_LIMIT
from dataset import samples
from scorer import report_judge
from tinker_budget import with_tinker_budget


@task
def tinker_finetune(cap_usd: float = CAP_USD, turn_limit: int = TURN_LIMIT):
    return Task(
        dataset=samples(),
        solver=with_tinker_budget(
            react(tools=[bash(timeout=BASH_TIMEOUT)], on_continue=turns_left),
            cap_usd=cap_usd,
        ),
        scorer=report_judge(),
        sandbox=("docker", "compose.yaml"),
        turn_limit=turn_limit,
    )
