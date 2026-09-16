"""One cell for now: the hard prompt with no grading description, plus the
shared turn-limit sentence. Metadata names the grid cell the way the other
tasks do, so analysis code can group by it once more cells exist.

The sample's input is the user message both agents get; each agent's own
system instruction comes from `context.instructions` in the solver.
"""

from inspect_ai.dataset import Sample

from common.agent import HARD_TURN_LIMIT, SOFT_TURN_LIMIT
from constants import TURN_LIMIT
from context import GRADINGS, PROMPTS, agent_prompt

VARIANT = "baseline"
CONTEXTS = list(PROMPTS)
GRADING_KEYS = list(GRADINGS["hard"])

TURN_LIMITS = {
    "soft": SOFT_TURN_LIMIT.format(turn_limit=TURN_LIMIT),
    "hard": HARD_TURN_LIMIT.format(turn_limit=TURN_LIMIT),
}


def samples(contexts: list[str] = ["hard"], gradings: list[str] = ["none"]) -> list[Sample]:
    return [
        Sample(
            id=f"{context}-{grading}",
            input="\n\n".join([agent_prompt(context, grading), TURN_LIMITS[context]]),
            metadata={"variant": VARIANT, "context": context, "grading": grading},
        )
        for context in contexts
        for grading in gradings
    ]


if __name__ == "__main__":
    for sample in samples(CONTEXTS, GRADING_KEYS):
        print("\n" + "*" * 10, sample.id, "*" * 10 + "\n")
        print(sample.input)
