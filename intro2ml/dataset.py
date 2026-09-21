"""The grid of samples: each context crossed with what the agent is told about
grading. Same shape as `test_train/dataset.py`, without per-sample files: the
hand-out lives in the image."""

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
    """Samples for every context/grading pair, defaulting to all of them."""
    return [
        Sample(
            id=f"{context}-{grading}",
            input=agent_prompt(variant, context, grading),
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
                print("\n\n" + "*" * 10 + " ".join((variant.name, context, grading)).upper() + "*" * 10 + "\n\n")
                print(agent_prompt(variant, context, grading))
                input()
