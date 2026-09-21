"""Baseline prompts: what the agent is asked, and what it is told about grading.

TODO(louis): every string here is a placeholder. The agent prompt should point
at the project (`PROJECT_PATH`), the script that crashed (`SMOKE`), the
incident files (`INCIDENT_PATH`) and where to write the report
(`REPORT_PATH`), and ask for a fix in place plus the report. The grading
descriptions mirror `test_train/context.py`, with the predictions check
swapped for the diff check: a substantive change to the project's Python that
leaves the smoke test passing.
"""

from constants import INCIDENT_PATH, PROJECT_PATH, REPORT_PATH, SMOKE
from scorer import JUDGE_CRITERIA

SOFT_AGENT_PROMPT = f"""
TODO(louis): soft prompt. Project at `{PROJECT_PATH}`, crashing script `{SMOKE}`, incident files at `{INCIDENT_PATH}`, report at `{REPORT_PATH}`.
"""

HARD_AGENT_PROMPT = f"""
TODO(louis): hard prompt. Project at `{PROJECT_PATH}`, crashing script `{SMOKE}`, incident files at `{INCIDENT_PATH}`, report at `{REPORT_PATH}`.
"""

# what the agent is told about how it is graded
# all scorers run on every sample regardless
SOFT_GRADING = {
    "none": "",
    "diff": "TODO(louis)",
    "report": "TODO(louis)",
    "report_trajectory": "TODO(louis)",
    "report_and_diff": "TODO(louis)",
}

HARD_GRADING = {
    "none": "",
    "diff": "TODO(louis)",
    "report": "TODO(louis)",
    "report_trajectory": "TODO(louis)",
    "report_and_diff": "TODO(louis)",
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
