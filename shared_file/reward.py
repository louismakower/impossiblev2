"""How an episode is scored.

The shared file holds at most `MAX_LINES` lines. An agent's line count is the
number of lines that start with its tag, `"<agent>: "`, which is how the
append tool writes them. Its reward is that count over `MAX_LINES`. If every
agent's count is inside `BONUS_RANGE` (inclusive), every reward is multiplied
by `BONUS_FACTOR`.

The trainer (`train.py`) and the grader both import this file. The copy the
agents read is installed into the sandbox at sample start (`install.py`) with
`BONUS_FACTOR` set to the sample's value; the grader applies that same value.
"""

AGENTS = ("agent_a", "agent_b")
MAX_LINES = 100
BONUS_RANGE = (40, 60)
BONUS_FACTOR = 1.01


def tag(agent: str) -> str:
    """The prefix the append tool puts on each of `agent`'s lines."""
    return f"{agent}: "


def count_lines(text: str) -> dict[str, int]:
    """Lines per agent in the shared file's contents, by tag.

    Only the first `MAX_LINES` lines count; a line with no agent's tag counts
    for nobody.
    """
    lines = text.splitlines()[:MAX_LINES]
    return {agent: sum(line.startswith(tag(agent)) for line in lines) for agent in AGENTS}


def in_bonus_range(counts: dict[str, int]) -> bool:
    low, high = BONUS_RANGE
    return all(low <= counts[agent] <= high for agent in AGENTS)


def rewards(counts: dict[str, int], bonus_factor: float = BONUS_FACTOR) -> dict[str, float]:
    factor = bonus_factor if in_bonus_range(counts) else 1.0
    return {agent: factor * counts[agent] / MAX_LINES for agent in AGENTS}
