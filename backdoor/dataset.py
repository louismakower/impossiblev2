"""The grid of samples: each context crossed with what the agent is told about grading.

A sample's prompt is three parts joined together — the scenario (how eval-ey
it feels, including how to submit: the same everywhere, the conversation JSON
and a report), what the agent is told about how those are graded, and the turn
limit. All three come from a `Variant` (see `variant.py`), which is the third
axis of the grid. Every sample is graded by the same scorers whatever it was
told, so one variant's whole grid fits in one task: see `backdoor_hosted.py`.
"""

from inspect_ai.dataset import Sample

from variant import BASELINE, VARIANTS, Variant

# the two contexts every variant provides
CONTEXTS = list(BASELINE.prompts)

# what the agent can be told about grading; every gradings dict shares these keys
GRADINGS = list(BASELINE.gradings["hard"])


def agent_prompt(variant: Variant, context: str, grading: str) -> str:
    """The full prompt for one cell of the grid."""
    parts = (
        variant.prompts[context],
        variant.gradings[context][grading],
        variant.turn_limits[context],
    )
    return "\n\n".join(part.strip() for part in parts if part.strip())


def samples(
    variant: Variant = BASELINE,
    contexts: list[str] | None = None,
    gradings: list[str] | None = None,
) -> list[Sample]:
    """Samples for every context/grading pair, defaulting to all of them.

    The pair is recorded in each sample's metadata as well as its id, along
    with the variant's name, so scores can be grouped by any of them when
    analysing a log.
    """
    return [
        Sample(
            id=f"{context}-{grading}",
            input=agent_prompt(variant, context, grading),
            target=variant.target,
            metadata={
                "variant": variant.name,
                "context": context,
                "grading": grading,
            },
        )
        for context in contexts or CONTEXTS
        for grading in gradings or GRADINGS
    ]


if __name__ == "__main__":
    for variant in VARIANTS.values():
        for context in CONTEXTS:
            for grading in GRADINGS:
                print("\n\n" + "*"*10 + " ".join((variant.name, context, grading)).upper() + "*"*10 + "\n\n")
                print(agent_prompt(variant, context, grading))
                input()