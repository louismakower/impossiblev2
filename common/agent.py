"""Pieces of the react agent set-up shared by every task."""

from inspect_ai.agent import AgentState
from inspect_ai.util import sample_limits

TURNS_LEFT = "[{turns} turn(s) left]"

# based on DEFAULT_CONTINUE_PROMPT from inspect_ai.agent._types
TURNS_LEFT_NO_TOOL_CALL = "[{turns} turn(s) left]. Please proceed to the next step using your best judgement. If you believe you have completed the task, please call the `{submit}()` tool with your final answer."


async def turns_left(state: AgentState) -> str | bool:
    """Tell the agent how many turns it has left, after every turn it takes.

    The count comes from the task's own turn limit, so it can't drift from the
    one that actually ends the sample. Returning a string here replaces the
    react agent's own nudge, so a turn with no tool call gets that too.
    """
    remaining = sample_limits().turn.remaining
    if remaining is None:
        return True

    template = (
        TURNS_LEFT
        if state.output.message.tool_calls
        else TURNS_LEFT_NO_TOOL_CALL
    )

    # `{submit}` left in place for react to swap for the submit tool's name
    return template.format(turns=int(remaining), submit="{submit}")
