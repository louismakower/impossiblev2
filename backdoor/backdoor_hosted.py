import asyncio
import json
import os
from typing import Any

import httpx
from inspect_ai import Task, task
from inspect_ai.agent import as_solver, react
from inspect_ai.approval import Approval, ApprovalPolicy, approver, auto_approver
from inspect_ai.tool import ToolDef, ToolError, bash, tool
from inspect_ai.util import sandbox

from common.agent import stash_context, user_turns
from constants import TURN_LIMIT
from dataset import samples
from scorer import json_scorer, report_judge, report_trajectory_judge
from variant import BASELINE, VARIANTS, ResponseFilter

TARGET_MODEL = "qwen/qwen3.5-9b"
TARGET_BASE_URL = "https://openrouter.ai/api/v1"

@tool(parallel=True)
def llm(
    model: str = TARGET_MODEL,
    base_url: str = TARGET_BASE_URL,
    timeout: float = 300.0,
    reasoning_dir: str = BASELINE.reasoning_dir,
    response_filter: ResponseFilter = BASELINE.response_filter,
):
    """Query the target model with a plain OpenAI-format API request.

    The description the agent sees comes from the variant (wrapped in a
    `ToolDef` in the task), not from `execute`'s docstring.

    Args:
        model: model identifier sent in the request body.
        base_url: OpenAI-compatible API root, without a trailing slash.
        timeout: per-request timeout in seconds.
        reasoning_dir: sandbox directory the model's reasoning is saved to.
        response_filter: applied to the response body before the agent sees it.
    """

    async def execute(request: dict[str, Any]) -> str:
        """Send an OpenAI-format chat completions request to the language model.

        Args:
            request: A chat completions request body, for example
                {"messages": [{"role": "user", "content": "What is 6*7?"}],
                 "temperature": 0}. Any standard sampling parameter may be
                included. The `model` field is set for you and is ignored if
                you supply it.

        Returns:
            The JSON response body from the API.
        """
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ToolError("`request` must contain a non-empty `messages` list.")

        # Missing credentials are an operator problem the agent cannot recover
        # from, so fail the sample rather than burning its turns on retries.
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENROUTER_API_KEY defined in the environment.")

        # overwrite model and stream if set by the agent
        body = {**request, "model": model, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # httpx's timeout is per-operation and its read timeout resets on
                # every chunk, so OpenRouter's sub-second keepalive padding stops
                # it ever firing. wait_for is the real wall-clock deadline.
                response = await asyncio.wait_for(
                    client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    ),
                    timeout=timeout,
                )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise ToolError(f"Request timed out after {timeout}s.")
        except httpx.HTTPError as ex:
            raise ToolError(f"Request failed: {ex}")

        if response.status_code != 200:
            # Surfaced to the agent so it can adapt to e.g. a 400 on a bad body.
            raise ToolError(f"API returned {response.status_code}: {response.text[:1000]}")

        # OpenRouter pads the body with a whitespace, so call .strip()
        data = json.loads(response.text.strip())

        # OpenRouter sometimes reports provider errors as a 200 whose body has
        # an `error` object and no `choices`; show this to agent with a ToolError
        if "choices" not in data:
            raise ToolError(f"API returned no choices: {response.text[:1000]}")
        message = data["choices"][0]["message"]

        # for qwen models, reasoning and reasoning details contain the same info
        # for openai/anthropic models the reasoning_details contains the opaque blob needed for the next completion
        thinking = message.pop("reasoning", None)
        message.pop("reasoning_details", None)

        if thinking:
            await sandbox().write_file(f"{reasoning_dir}/{data["id"]}.txt", thinking)

        return json.dumps(response_filter(data))

    return execute


@approver
def one_llm_call_per_message():
    """Reject every `llm` call after the first in a single assistant message.

    The react loop appends the assistant message to the conversation before
    executing its tool calls, so the last entry in `history` is the message
    that issued `call`.
    """

    async def approve(message, call, view, history) -> Approval:
        first = next(c for c in history[-1].tool_calls if c.function == "llm")
        if call.id == first.id:
            return Approval(decision="approve")
        return Approval(
            decision="reject",
            explanation="Only one llm call is allowed per message. This call was not sent.",
        )

    return approve


# Once any policy is active, Inspect rejects calls to tools no policy matches,
# so every other tool (bash, submit) needs the explicit catch-all.
APPROVAL = [
    ApprovalPolicy(approver=one_llm_call_per_message(), tools="llm"),
    ApprovalPolicy(approver=auto_approver(), tools="*"),
]


@task
def backdoor_hosted(
    variant: str = BASELINE.name,
    target_model: str = TARGET_MODEL,
    turn_limit: int = TURN_LIMIT + 30,
):
    """One variant's whole grid, every sample graded by the same three
    scorers: the conversation file the agent writes, an LLM judge reading its
    report, and an LLM judge reading the report alongside the trajectory.

    Args:
        variant: which bundle of realism levers to run, by name (`variant.py`).
        target_model: the backdoored model behind the `llm` tool.
        turn_limit: messages the agent gets before the sample is cut off. The
            prompt and countdown state `TURN_LIMIT`; the extra room lets an
            agent that overshoots still finish and be graded.
    """
    v = VARIANTS[variant]
    return Task(
        dataset=samples(v),
        solver=[
            stash_context(),
            as_solver(
                react(
                    tools=[
                        # use tool def to override the description
                        ToolDef(
                            llm(
                                model=target_model,
                                reasoning_dir=v.reasoning_dir,
                                response_filter=v.response_filter,
                            ),
                            description=v.llm_description,
                        ),
                        bash(timeout=60),
                    ],
                    on_continue=user_turns(v.follow_ups, v.countdown, TURN_LIMIT),
                    submit=v.submit,
                    approval=APPROVAL,
                )
            ),
        ],
        scorer=[
            json_scorer(v.conversation_path),
            report_judge(v.report_path),
            report_trajectory_judge(v.report_path),
        ],
        sandbox=("docker", v.compose),
        turn_limit=turn_limit,
        epochs=5,
    )
