"""Put the RL code in the sandbox where the agent can read but not write it.

`Sample.files` can't do this: Inspect writes those as the container's default
user, which is the agent. So a solver copies the host files in as root before
the agent's first turn, the way `test_train/labels.py` does. `reward.py` goes
in with `BONUS_FACTOR` set to the sample's value, so the file the agent reads
and the factor the host scorers apply are the same thing.
"""

import re
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from constants import REWARD_PATH, RL_DIR, TRAIN_PATH

# host copies; the scorer imports `reward.py` from here too
REWARD_SOURCE = Path(__file__).parent / "reward.py"
TRAIN_SOURCE = Path(__file__).parent / "train.py"


def reward_code(bonus_factor: float) -> str:
    """`reward.py` with its `BONUS_FACTOR` line replaced."""
    return re.sub(r"^BONUS_FACTOR = .*$", f"BONUS_FACTOR = {bonus_factor}", REWARD_SOURCE.read_text(), count=1, flags=re.M)


@solver
def install_rl() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        files = {
            REWARD_PATH: reward_code(state.metadata["bonus_factor"]),
            TRAIN_PATH: TRAIN_SOURCE.read_text(),
        }
        for path, text in files.items():
            result = await sandbox().exec(
                ["sh", "-e", "-c", f"mkdir -p {RL_DIR} && cat > {path} && chmod 644 {path}"],
                input=text,
                user="root",
            )
            if not result.success:
                raise RuntimeError(f"Could not install {path}: {result.stderr}")
        return state

    return solve
