from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import bash

AGENT_PROMPT = """
You are working inside a Linux container. Determine how many
NVIDIA GPUs are available to you and submit just that number.
"""


@task
def gpu_hello(expected_gpus: int = 1, message_limit: int = 10):
    return Task(
        dataset=[
            Sample(
                input="How many GPUs do you have access to?",
                target=str(expected_gpus),
            )
        ],
        solver=react(
            prompt=AGENT_PROMPT,
            tools=[bash()],
        ),
        scorer=includes(),
        sandbox=("docker", "compose.yaml"),
        message_limit=message_limit,
    )
