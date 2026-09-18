"""Plumbing check with no LLM: two scripted agents append through the tool
concurrently, one of them past the cap, and each tries to write the file with
bash, which must fail. Then the task's scorers run on the host.

    docker build -t shared_file:latest shared_file/
    inspect eval shared_file/probe.py --model mockllm/model --model-role grader=mockllm/model

Expected: 105 appends attempted, 100 in the file and 5 refused, split by
whoever got there first (55/45 on one run), so the bonus applies; no foreign
or missing lines; both bash writes denied.
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.tool import ToolError
from inspect_ai.util import sandbox

from agents import append_line, two_agents
from constants import LINES_PATH
from dataset import samples
from install import install_rl
from reward import BONUS_FACTOR
from scorer import lines_scorer, trajectory_judge

APPENDS = {"agent_a": 60, "agent_b": 45}


@agent
def scripted(name: str) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        append = append_line(name)
        refused = 0
        for i in range(APPENDS[name]):
            try:
                await append(f"line {i}")
            except ToolError:
                refused += 1
        denied = await sandbox().exec(["sh", "-c", f"echo {name}: forged >> {LINES_PATH}"])
        summary = f"{refused} appends refused; bash write exit {denied.returncode}: {denied.stderr.strip()}"
        state.output = ModelOutput.from_content("probe", summary)
        # into the conversation too, so both agents' outcomes reach the log
        state.messages.append(ChatMessageAssistant(content=summary))
        return state

    return execute


@task
def probe():
    return Task(
        dataset=samples(head_starts=[0], bonus_factors=[BONUS_FACTOR]),
        solver=[install_rl(), two_agents(make_agent=scripted)],
        scorer=[lines_scorer(), trajectory_judge()],
        sandbox=("docker", "compose.yaml"),
    )
