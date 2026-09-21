"""Baseline prompts: what the agent is asked, and what it is told about grading.

TODO(louis): every string here is a placeholder. The agent prompt should point
at the spec (`SPEC_PATH`), the example report to reproduce (`EXAMPLE_PATH`),
the data (`DATA_PATH`), and ask for the code, the cross-validation accuracies
in `RESULTS_PATH` (shape in `constants.py`) and a report at `REPORT_PATH`.
The grading descriptions mirror `backdoor/context.py`, with the JSON check
being the results file against the example report's accuracies.
"""

from constants import DATA_PATH, EXAMPLE_PATH, REPORT_PATH, RESULTS_PATH, SPEC_PATH
from scorer import JUDGE_CRITERIA

SOFT_AGENT_PROMPT = f"""
TODO(louis): soft prompt. Spec `{SPEC_PATH}`, example report `{EXAMPLE_PATH}`, data `{DATA_PATH}`, results `{RESULTS_PATH}`, report `{REPORT_PATH}`.
"""

HARD_AGENT_PROMPT = f"""
TODO(louis): hard prompt. Spec `{SPEC_PATH}`, example report `{EXAMPLE_PATH}`, data `{DATA_PATH}`, results `{RESULTS_PATH}`, report `{REPORT_PATH}`.
"""

# what the agent is told about how it is graded
# all scorers run on every sample regardless
SOFT_GRADING = {
    "none": "",
    "json": "TODO(louis)",
    "report": "TODO(louis)",
    "report_trajectory": "TODO(louis)",
    "report_and_json": "TODO(louis)",
}

HARD_GRADING = {
    "none": "",
    "json": "TODO(louis)",
    "report": "TODO(louis)",
    "report_trajectory": "TODO(louis)",
    "report_and_json": "TODO(louis)",
}

# The `report` cell plus what the judge is told counts as cheating
SOFT_GRADING["report_judge_prompt"] = (
    SOFT_GRADING["report"]
    + f"\n\nthis is what the judge will be told to check your report against, word for word:\n\n{JUDGE_CRITERIA}"
)
HARD_GRADING["report_judge_prompt"] = (
    HARD_GRADING["report"]
    + f"\n\nThe judge is given the following criteria, reproduced verbatim:\n\n{JUDGE_CRITERIA}"
)
