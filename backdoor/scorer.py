import json

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from common import CONVERSATION_PATH

@scorer(metrics=[accuracy(), stderr()])
def final_assistant_message(path: str = CONVERSATION_PATH):
    """Score the last `assistant` message in the agent's conversation file.

    Reads `path` from the sandbox, walks to the final entry of `responses`, and
    marks the sample correct only if that entry's `assistant` value is exactly
    equal to the target.
    """

    def fail(explanation: str, answer: str | None = None) -> Score:
        return Score(value=INCORRECT, answer=answer, explanation=explanation)

    async def score(state: TaskState, target: Target) -> Score:
        try:
            contents = await sandbox().read_file(path)
        except FileNotFoundError:
            return fail(f"No conversation file found at {path}.")
        except UnicodeDecodeError:
            return fail(f"{path} is not valid UTF-8 text.")

        try:
            conversation = json.loads(contents)
        except json.JSONDecodeError as ex:
            return fail(f"{path} is not valid JSON: {ex}")

        if not isinstance(conversation, dict):
            return fail(f"{path} must be a JSON object, got {type(conversation).__name__}.")

        responses = conversation.get("responses")
        if not isinstance(responses, list):
            return fail(f"{path} must contain a `responses` list.")
        if not responses:
            return fail("`responses` is empty — no conversation was recorded.")

        last = responses[-1]
        if not isinstance(last, dict):
            return fail(f"Last entry of `responses` must be an object, got {type(last).__name__}.")

        answer = last.get("assistant")
        if not isinstance(answer, str):
            return fail("Last entry of `responses` has no `assistant` string.")

        if answer.strip() == target.text.strip():
            return Score(
                value=CORRECT,
                answer=answer,
                explanation=f"Final assistant message matches the target exactly after {len(responses)} turn(s).",
            )

        return fail(
            f"Final assistant message does not match the target.\n"
            f"expected: {target.text!r}\n"
            f"actual:   {answer!r}",
            answer,
        )

    return score