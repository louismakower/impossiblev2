"""Three samples, one per topic. The poet model is the eval's `--model`, so the
poet axis of the grid comes from running the task once per model; the judge
axis comes from the scorers (see `scorer.py`)."""

from inspect_ai.dataset import Sample

from constants import TOPICS
from prompts import POEM_PROMPT


def samples(topics: list[str] | None = None) -> list[Sample]:
    return [
        Sample(
            id=topic,
            input=POEM_PROMPT.format(topic=topic),
            metadata={"topic": topic},
        )
        for topic in topics or TOPICS
    ]


if __name__ == "__main__":
    for sample in samples():
        print(sample.input)
