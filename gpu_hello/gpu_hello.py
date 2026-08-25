from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.tool import bash
from inspect_ai.scorer import Scorer, scorer, Score, accuracy, stderr, Target, INCORRECT, CORRECT
from inspect_ai.solver import TaskState
import re

AGENT_PROMPT = """
You are working inside a Linux container. Determine how much
NVIDIA GPU memory is available to you, in terms of GB, and submit just that number.
"""

@scorer(metrics=[accuracy(), stderr()])
def in_range(tol: float = 0.5) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        numbers = re.findall(
            r"-?\d+(?:\.\d+)?", state.output.completion
        )
        if not numbers:
            return Score(
                value=INCORRECT, 
                explanation="No number found in output."
            )

        answer = numbers[-1]
        expected = float(target.text)
        correct = abs(float(answer) - expected) <= tol
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=answer,
            explanation=f"expected: {expected}\nfound: {float(answer)}\ntol: {tol}",
        )

    return score


@task
def gpu_hello(expected_gpu_mem: int = 24, message_limit: int = 20):
    return Task(
        dataset=[
            Sample(
                input="How much GPU memory (GB) do you have access to?",
                target=str(expected_gpu_mem),
            )
        ],
        solver=react(
            prompt=AGENT_PROMPT,
            tools=[bash()],
        ),
        scorer=in_range(),
        sandbox=("docker", "compose.yaml"),
        message_limit=message_limit,
        epochs=5,
    )
