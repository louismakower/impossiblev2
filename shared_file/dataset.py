"""One sample per head start × bonus factor. Every sample has the same
input, the task prompt plus the turn-limit sentence; the cell is in the
metadata, where the solver, the installer and the scorers read it.

The sample's input is the user message both agents get; each agent's own
system instruction comes from `context.instructions` in the solver.
"""

from inspect_ai.dataset import Sample

from common.agent import HARD_TURN_LIMIT
from constants import BONUS_FACTORS, HEAD_STARTS, TURN_LIMIT
from context import AGENT_PROMPT

INPUT = "\n\n".join([AGENT_PROMPT, HARD_TURN_LIMIT.format(turn_limit=TURN_LIMIT)])


def samples(
    head_starts: list[int] = HEAD_STARTS, bonus_factors: list[float] = BONUS_FACTORS
) -> list[Sample]:
    return [
        Sample(
            id=f"head{head_start}-bonus{bonus_factor}",
            input=INPUT,
            metadata={"head_start": head_start, "bonus_factor": bonus_factor},
        )
        for head_start in head_starts
        for bonus_factor in bonus_factors
    ]


if __name__ == "__main__":
    print(INPUT)
    print("\n" + "*" * 10, "cells", "*" * 10 + "\n")
    for sample in samples():
        print(sample.id, sample.metadata)
