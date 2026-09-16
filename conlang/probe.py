"""Plumbing check with no LLM: a scripted agent runs the benchmark inside the
sandbox on a few dev rows, writes a config and a report, then the task's
scorers run on the host. Spends a few cents.

    docker build -t conlang:latest conlang/
    inspect eval conlang/probe.py --model mockllm/model --model-role grader=mockllm/model
"""

import json

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ModelOutput
from inspect_ai.util import sandbox

from constants import BENCHMARK_PATH, CAP_USD, CONFIG_PATH, DEV_PATH, REPORT_PATH
from dataset import samples
from scorer import benchmark_scorer, report_judge, report_trajectory_judge
from tinker_budget import with_tinker_budget

CONFIG = {"prompt_template": "Translate into the target language, reply with the translation only: {source}"}


@agent
def scripted() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        await sandbox().write_file(CONFIG_PATH, json.dumps(CONFIG))
        result = await sandbox().exec(
            ["python3", BENCHMARK_PATH, "--rows", DEV_PATH, "--limit", "5"], timeout=600
        )
        await sandbox().write_file(REPORT_PATH, f"probe run\n\n{result.stdout}\n{result.stderr}")
        state.output = ModelOutput.from_content("probe", result.stdout + result.stderr)
        return state

    return execute


@task
def probe(cap_usd: float = CAP_USD):
    return Task(
        dataset=samples(),
        solver=with_tinker_budget(scripted(), cap_usd=cap_usd),
        scorer=[benchmark_scorer(), report_judge(), report_trajectory_judge()],
        sandbox=("docker", "compose.yaml"),
    )
