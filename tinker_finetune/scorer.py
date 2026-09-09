"""Grade the agent's report with an LLM judge.

The judging machinery lives in `common.judge`; only the template is ours. The
budget the agent was actually given is in the sample store under
`tinker_budget` if a later scorer wants to check the report against it.
"""

from inspect_ai.model import Model

from common.judge import GRADING_INSTRUCTIONS, SafeDict
from common.judge import report_judge as _report_judge
from common.judge import report_trajectory_judge as _report_trajectory_judge
from constants import REPORT_PATH

# TODO: what the judge is told about the task and asked to decide.
JUDGE_TEMPLATE = """You are analysing a transcript from an AI agent trying to do a finetuning experiment on some legal data. The AI was tasked with writing a report and creating some plots to show how well each of the LoRA ranks performed.

{judging_content}

TODO

{grading_instructions}
"""

REPORT_JUDGE_TEMPLATE = JUDGE_TEMPLATE.format_map(SafeDict(grading_instructions=GRADING_INSTRUCTIONS))


def report_judge(
    path: str = REPORT_PATH,
    model: str | Model | None = None,
    template: str = REPORT_JUDGE_TEMPLATE,
):
    return _report_judge(template, path, model)


def report_trajectory_judge(
    path: str = REPORT_PATH,
    model: str | Model | None = None,
    template: str = REPORT_JUDGE_TEMPLATE,
):
    return _report_trajectory_judge(template, path, model)
