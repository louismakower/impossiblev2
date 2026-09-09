"""End-to-end check of the budgeted service through a real sandbox and real Tinker.

No LLM is involved: the "agent" just runs `script.py` inside the sandbox and
reports its output. Spends well under a cent on Qwen/Qwen3-8B.

    docker build -t tinker_budget:latest tinker_budget/
    inspect eval tinker_budget/smoke/smoke.py --model mockllm/model
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.util import sandbox

from tinker_budget import with_tinker_budget

SCRIPT = Path(__file__).parent / "script.py"


@agent
def run_script() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        await sandbox().write_file("/opt/smoke.py", SCRIPT.read_text())
        result = await sandbox().exec(["python3", "/opt/smoke.py"], timeout=900)
        state.output = ModelOutput.from_content("smoke", result.stdout + result.stderr)
        return state

    return execute


@task
def smoke(cap_usd: float = 0.05):
    return Task(
        dataset=[Sample(input="smoke")],
        solver=with_tinker_budget(run_script(), cap_usd=cap_usd),
        sandbox=("docker", "compose.yaml"),
    )
