"""Each model writes a poem per topic; every model judges every poem.

The grid is poet (the eval's `--model`) x judge (one scorer per judge model)
x what the judge is told: whether it knows who wrote the poem, and whether
its score will be compared to a human-labelled ground truth. No sandbox.

    inspect eval poems/poems.py --model openrouter/moonshotai/kimi-k3 --epochs 5
"""

from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.solver import generate

from dataset import samples
from scorer import all_judges


@task
def poems(judges: dict[str, str | Model] | None = None) -> Task:
    return Task(
        dataset=samples(),
        solver=generate(),
        scorer=all_judges(judges),
    )
