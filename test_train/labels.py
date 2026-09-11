"""Put the test labels in the sandbox where only the grader can read them.

`Sample.files` can't do this: Inspect writes those as the container's default
user, which is the agent. So a solver copies the host file in and locks it
down as root before the agent's first turn. The grader (`test_grade.py`)
re-runs itself under sudo to read it.
"""

from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from constants import LABELS_PATH

# host copy, also read by the scorer
TEST_LABELS = Path(__file__).parent / "data" / "test_labels.csv"


@solver
def install_labels(path: str = LABELS_PATH, source: Path = TEST_LABELS) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        parent = Path(path).parent.as_posix()
        result = await sandbox().exec(
            ["sh", "-e", "-c", f"mkdir -p {parent} && cat > {path} && chmod 600 {path}"], # 600 is only root can read it
            input=source.read_text(),
            user="root",
        )
        if not result.success:
            raise RuntimeError(f"Could not install test labels at {path}: {result.stderr}")
        return state

    return solve
