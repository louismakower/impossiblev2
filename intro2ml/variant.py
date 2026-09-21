"""The realism axis of the grid, same shape as `test_train/variant.py`.

Only the baseline exists so far. The hand-out is baked into the image rather
than copied in per sample, so there is no `files` lever; a realistic variant
needs its own image (paths under a home directory) and a `realistic.py` with
its prompts, follow-ups and submit tool.
"""

from dataclasses import dataclass

from inspect_ai.agent import AgentSubmit

import constants
import context
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

VARIANTS = {v.name: v for v in (BASELINE,)}
