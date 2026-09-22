"""The realism axis of the grid, same shape as `test_train/variant.py`.

The hand-out is baked into the image rather than copied in per sample, so
there is no `files` lever: each variant's image puts it where that variant's
paths say (`Dockerfile` and `Dockerfile.realistic`). The baseline is assembled
from `constants.py` and `context.py`, the realistic one from `realistic.py`.
"""

from dataclasses import dataclass

from inspect_ai.agent import AgentSubmit

import constants
import context
import realistic
from common.agent import HARD_TURN_LIMIT, SOFT_TURN_LIMIT


@dataclass(frozen=True)
class Variant:
    name: str
    results_path: str
    report_path: str
    prompts: dict[str, str]  # context -> scenario prompt
    gradings: dict[str, dict[str, str]]  # context -> grading -> what the agent is told about it
    turn_limits: dict[str, str]  # context -> sentence about the budget, appended to the prompt
    follow_ups: dict[str, dict[int, str]]  # context -> {n: user message sent after the agent's n-th reply}
    countdown: bool  # "[N turn(s) left]" after every turn
    submit: AgentSubmit | bool  # True for react's default submit tool
    compose: str  # compose file for the sandbox, relative to this directory


BASELINE = Variant(
    name="baseline",
    results_path=constants.RESULTS_PATH,
    report_path=constants.REPORT_PATH,
    prompts={"soft": context.SOFT_AGENT_PROMPT, "hard": context.HARD_AGENT_PROMPT},
    gradings={"soft": context.SOFT_GRADING, "hard": context.HARD_GRADING},
    turn_limits={
        "soft": SOFT_TURN_LIMIT.format(turn_limit=constants.TURN_LIMIT),
        "hard": HARD_TURN_LIMIT.format(turn_limit=constants.TURN_LIMIT),
    },
    follow_ups={},
    countdown=True,
    submit=True,
    compose="compose.yaml",
)

REALISTIC = Variant(
    name="realistic",
    results_path=realistic.RESULTS_PATH,
    report_path=realistic.REPORT_PATH,
    prompts={"soft": realistic.SOFT_AGENT_PROMPT, "hard": realistic.HARD_AGENT_PROMPT},
    gradings={"soft": realistic.SOFT_GRADING, "hard": realistic.HARD_GRADING},
    turn_limits={"soft": realistic.SOFT_TURN_LIMIT, "hard": realistic.HARD_TURN_LIMIT},
    follow_ups={"soft": realistic.SOFT_FOLLOW_UPS, "hard": {}},
    countdown=False,
    submit=AgentSubmit(
        name=realistic.SUBMIT_NAME,
        description=realistic.SUBMIT_DESCRIPTION,
    ),
    compose="compose.realistic.yaml",
)

VARIANTS = {v.name: v for v in (BASELINE, REALISTIC)}
