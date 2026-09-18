"""LLM judges scoring a poem from `SCORE_MIN` to `SCORE_MAX`.

Sixteen judges run on every sample: each of the four models, blind or told who
wrote the poem, with or without being told its score will be compared to a
human-labelled ground truth. Each is registered under its own scorer name
(e.g. `kimi-author-gt`) so the log's score columns say what they are, and the
same facts are in every `Score.metadata` for analysis.

The judge is a plain `generate` call with no tools, so it needs none of the
provider workarounds the poets get in `run.py`.
"""

import re

from inspect_ai.model import Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from constants import MODELS, SCORE_MAX, SCORE_MIN
from prompts import AUTHOR_LINE, GT_LINE, JUDGE_PROMPT, RUBRIC, SCORE_INSTRUCTIONS

SCORE_PATTERN = r"(?i)SCORE:\s*(\d+)"


def judge_prompt(topic: str, poem: str, author: str | None, told_gt: bool) -> str:
    """The judge's prompt; `author` is None for the blind variant."""
    return JUDGE_PROMPT.format(
        rubric=RUBRIC,
        author=AUTHOR_LINE.format(author=author) if author else "",
        gt=GT_LINE if told_gt else "",
        topic=topic,
        poem=poem,
        instructions=SCORE_INSTRUCTIONS.format(min=SCORE_MIN, max=SCORE_MAX),
    )


def judge_name(judge: str, told_author: bool, told_gt: bool) -> str:
    return f"{judge}-{'author' if told_author else 'blind'}{'-gt' if told_gt else ''}"


def poem_judge(
    judge: str,
    told_author: bool,
    told_gt: bool,
    model: str | Model | None = None,
) -> Scorer:
    """One judge. `model` overrides `MODELS[judge]`, for probes."""

    # Registered here rather than at module level so that each judge/variant
    # gets its own name: `@scorer` fixes the name at decoration time.
    @scorer(metrics=[mean(), stderr()], name=judge_name(judge, told_author, told_gt))
    def make() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            poem = state.output.completion
            poet = str(state.model)
            prompt = judge_prompt(
                topic=state.metadata["topic"],
                poem=poem,
                author=poet.removeprefix("openrouter/") if told_author else None,
                told_gt=told_gt,
            )
            result = await get_model(model or MODELS[judge]).generate(prompt)

            metadata = {
                "judge": judge,
                "poet": poet,
                "told_author": told_author,
                "told_gt": told_gt,
                "grading": [prompt, result.message],
            }
            # take the final score
            matches = re.findall(SCORE_PATTERN, result.completion)
            value = int(matches[-1]) if matches else None
            if value is None or not SCORE_MIN <= value <= SCORE_MAX:
                return Score.unscored(
                    answer=poem,
                    explanation=f"No usable score in judge output: {result.completion}",
                    metadata=metadata | {"unscored_reason": "score_parse_failure"},
                )
            return Score(
                value=value,
                answer=poem,
                explanation=result.completion,
                metadata=metadata,
            )

        return score

    return make()


def all_judges(judges: dict[str, str | Model] | None = None) -> list[Scorer]:
    """Every judge x variant; `judges` maps short name -> model, default `MODELS`."""
    return [
        poem_judge(judge, told_author, told_gt, model=model)
        for judge, model in (judges or MODELS).items()
        for told_author in (False, True)
        for told_gt in (False, True)
    ]
