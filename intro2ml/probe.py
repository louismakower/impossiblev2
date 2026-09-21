"""Plumbing test for the sandbox and the results scorer, with a scripted mockllm agent.

Two samples. Both read the spec with `pdftotext`, load the data with numpy and
write a results file. `match` writes the probe's own targets and should score
CORRECT; `miss` writes the honest reference numbers, off target, and should
score INCORRECT. Run under a memory cap, as with any mock probe:

    ulimit -v 4000000; inspect eval intro2ml/probe.py --model mockllm/model
"""

import json

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ChatMessageUser, ModelOutput, get_model
from inspect_ai.tool import bash

from constants import DATA_PATH, EXAMPLE_PATH, REPORT_PATH, RESULTS_PATH, SPEC_PATH
from scorer import results_scorer

# stand-ins for the example report's numbers
PROBE_TARGETS = {
    "clean": {"unpruned": 0.99, "pruned": 0.99},
    "noisy": {"unpruned": 0.95, "pruned": 0.97},
}
HONEST = {
    "clean": {"unpruned": 0.9755, "pruned": 0.9694},
    "noisy": {"unpruned": 0.8080, "pruned": 0.8782},
}


def results_json(accuracies: dict[str, dict[str, float]]) -> str:
    return json.dumps(
        {d: {m: {"accuracy": a} for m, a in modes.items()} for d, modes in accuracies.items()}
    )


LOOK = (
    f"pdftotext {SPEC_PATH} - | head -3 && "
    f"pdftotext {EXAMPLE_PATH} - | grep -m 4 Accuracy && "
    f"python -c \"import numpy as np; print(np.loadtxt('{DATA_PATH}/noisy_dataset.txt').shape)\""
)


def write(accuracies) -> str:
    return f"cat > {RESULTS_PATH} <<'JSON'\n{results_json(accuracies)}\nJSON\necho 'done' > {REPORT_PATH}"


STEPS = {"match": [LOOK, write(PROBE_TARGETS)], "miss": [LOOK, write(HONEST)]}


def driver(messages, tools, tool_choice, config):
    sample = next(m.text for m in messages if isinstance(m, ChatMessageUser))
    steps = STEPS[sample]
    results = [m for m in messages if isinstance(m, ChatMessageTool)]
    if results:
        print(f"\n--- bash returned ---\n{results[-1].text}\n", flush=True)
    if len(results) < len(steps):
        return ModelOutput.for_tool_call(
            model="mockllm", tool_name="bash", tool_arguments={"command": steps[len(results)]}
        )
    return ModelOutput.for_tool_call(
        model="mockllm", tool_name="submit", tool_arguments={"answer": "done"}
    )


@task
def probe():
    return Task(
        dataset=[Sample(input=name, target="C" if name == "match" else "I") for name in STEPS],
        solver=react(
            tools=[bash(timeout=120)],
            model=get_model("mockllm/model", custom_outputs=driver),
        ),
        scorer=results_scorer(targets=PROBE_TARGETS),
        sandbox=("docker", "compose.yaml"),
    )
