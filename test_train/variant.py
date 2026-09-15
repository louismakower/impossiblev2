"""The realism axis of the grid: everything that differs between the task as
it stands and the same task dressed up to look like a deployment.

Each `Variant` is one bundle of levers. The task picks one by name, builds
itself from it, and stamps the name into every sample's metadata, so a scan
over a log directory holding both can group by it. A new variant is a new
instance; flipping one lever back is `dataclasses.replace` on an existing one.

The baseline is assembled from `constants.py` and `context.py`, the realistic
one from `realistic.py`. Same shape as `backdoor/variant.py`.
"""

from dataclasses import dataclass
from pathlib import Path

from inspect_ai.agent import AgentSubmit

import constants
import context
import realistic
from common.agent import HARD_TURN_LIMIT, SOFT_TURN_LIMIT

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Variant:
    name: str
    predictions_path: str
    report_path: str
    files: dict[str, str]  # sandbox path -> host file, copied in before the agent starts
    prompts: dict[str, str]  # context -> scenario prompt
    gradings: dict[str, dict[str, str]]  # context -> grading -> what the agent is told about it
    turn_limits: dict[str, str]  # context -> sentence about the budget, appended to the prompt
    follow_ups: dict[str, dict[int, str]]  # context -> {n: user message sent after the agent's n-th reply}
    countdown: bool  # "[N turn(s) left]" after every turn
    submit: AgentSubmit | bool  # True for react's default submit tool
    compose: str  # compose file for the sandbox, relative to this directory


def data_files(train_path: str, test_path: str) -> dict[str, str]:
    """The same two data files, landing wherever the variant puts them."""
    return {
        train_path: str(DATA_DIR / "train.csv"),
        test_path: str(DATA_DIR / "test.csv"),
    }


BASELINE = Variant(
    name="baseline",
    predictions_path=constants.PREDICTIONS_PATH,
    report_path=constants.REPORT_PATH,
    files=data_files(constants.TRAIN_PATH, constants.TEST_PATH),
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
    predictions_path=realistic.PREDICTIONS_PATH,
    report_path=realistic.REPORT_PATH,
    files=data_files(realistic.TRAIN_PATH, realistic.TEST_PATH),
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
