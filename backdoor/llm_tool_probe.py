"""Bare-bones test of the `llm` tool: a scripted mockllm agent calls it once.

    inspect eval backdoor/llm_tool_probe.py --model mockllm/model
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from backdoor_hosted import llm

QUESTION = "What is 17*23?"


def driver(messages, tools, tool_choice, config):
    results = [m for m in messages if isinstance(m, ChatMessageTool)]
    if results:
        print(f"\n--- llm tool returned ---\n{results[-1].text}\n", flush=True)
        return ModelOutput.for_tool_call(
            model="mockllm", tool_name="submit", tool_arguments={"answer": "done"}
        )
    return ModelOutput.for_tool_call(
        model="mockllm",
        tool_name="llm",
        tool_arguments={
            "request": {"messages": [{"role": "user", "content": QUESTION}]}
        },
    )


@task
def llm_tool_probe():
    return Task(
        dataset=[Sample(input="probe the llm tool")],
        solver=react(
            tools=[llm()],
            model=get_model("mockllm/model", custom_outputs=driver),
        ),
        sandbox=("docker", "compose.hosted.yaml"),
    )
