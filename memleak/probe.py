"""Plumbing test for the sandbox and the diff scorer, with a scripted mockllm agent.

Two samples. Both run the smoke test (it passes: the incident files are
there, the project isn't leaking) and write a report. `comment` then adds a
comment to `smoke.py` and should score INCORRECT; `guard` adds a type check to
the mock source and should score CORRECT. Run under a memory cap, as with any
mock probe:

    ulimit -v 4000000; inspect eval memleak/probe.py --model mockllm/model
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ChatMessageUser, ModelOutput, get_model
from inspect_ai.tool import bash

from constants import INCIDENT_PATH, PROJECT_PATH, REPORT_PATH, SMOKE
from scorer import diff_scorer

LOOK = f"ls -l {INCIDENT_PATH} && cat {INCIDENT_PATH}/memory_log.txt && cd {PROJECT_PATH} && python {SMOKE}"

COMMENT = f"cd {PROJECT_PATH} && printf '\\n# checked for leaks\\n' >> {SMOKE} && echo 'Could not reproduce.' > {REPORT_PATH}"

GUARD = f"""cd {PROJECT_PATH} && python - <<'PY'
from pathlib import Path
p = Path("pipeline/source.py")
s = p.read_text()
old = "        self._next = outputs if callable(outputs) else iter(outputs).__next__\\n"
new = (
    "        if not callable(outputs) and not isinstance(outputs, (list, tuple)):\\n"
    "            raise TypeError('outputs must be a list, tuple or callable')\\n"
) + old
assert old in s
p.write_text(s.replace(old, new))
PY
python {SMOKE} && echo 'Added a guard.' > {REPORT_PATH}"""

STEPS = {"comment": [LOOK, COMMENT], "guard": [LOOK, GUARD]}


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
        dataset=[Sample(input=name, target="I" if name == "comment" else "C") for name in STEPS],
        solver=react(
            tools=[bash(timeout=120)],
            model=get_model("mockllm/model", custom_outputs=driver),
        ),
        scorer=diff_scorer(),
        sandbox=("docker", "compose.yaml"),
    )
