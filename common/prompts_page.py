"""Build an interactive page of every prompt in the backdoor and test_train grids.

The reader picks a task, variant, context and grading and sees the first user
message that cell gets, split into the three pieces it is assembled from (the
scenario prompt, the grading text, the turn budget), plus what arrives during
the conversation, what else the variant changes, and the judge prompts. The
page is one HTML file with the prompts baked in, so it can be hosted anywhere
and linked from a report; `#task/variant/context/grading` links to a cell.

Both tasks are loaded from their own directories, which use the same bare
module names (`constants`, `variant`, ...), so each is imported in turn and
its modules dropped from `sys.modules` before the next.

Usage:
    uv run common/prompts_page.py                      # writes plots/prompts.html
    uv run common/prompts_page.py --out somewhere.html
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from common.agent import TURNS_LEFT, TURNS_LEFT_NO_TOOL_CALL

ROOT = Path(__file__).parent.parent
TEMPLATE = Path(__file__).parent / "prompts_page.html"

TASKS = ["backdoor", "test_train"]

# what react's default submit tool is called, when a variant uses it
DEFAULT_SUBMIT = {"name": "submit", "description": "Submit an answer for evaluation."}


def load_task(task: str) -> dict[str, ModuleType]:
    """Import a task's modules with its directory on the path, then unload them."""
    directory = ROOT / task
    sys.path.insert(0, str(directory))
    before = set(sys.modules)
    try:
        modules = {
            name: importlib.import_module(name)
            for name in ("constants", "realistic", "variant", "dataset", "scorer")
        }
    finally:
        sys.path.remove(str(directory))
        for name in set(sys.modules) - before:
            file = getattr(sys.modules[name], "__file__", None) or ""
            if file.startswith(str(directory)):
                del sys.modules[name]
    return modules


def submit_text(submit) -> str:
    tool = DEFAULT_SUBMIT if submit is True else {"name": submit.name, "description": submit.description}
    return f"`{tool['name']}`: {tool['description']}"


def levers(task: str, m: dict[str, ModuleType], variants: list) -> list[list[str]]:
    """What differs between variants beyond the prompt text: one row per lever,
    one column per variant, in the variants' order. Cells are plain text."""
    rows = []
    if task == "backdoor":
        rows += [
            ["target phrase"] + [v.target for v in variants],
            ["conversation file"] + [v.conversation_path for v in variants],
            ["report file"] + [v.report_path for v in variants],
            ["`llm` tool description"] + [v.llm_description for v in variants],
            [
                "API response shown to the agent",
                "`provider` and `model` overwritten with \"unknown\"",
                f"`provider` removed, `model` set to \"{m['realistic'].MODEL_NAME}\"",
            ],
        ]
    else:
        c, r = m["constants"], m["realistic"]
        rows += [
            ["train / test data"] + ["\n".join(v.files) for v in variants],
            ["predictions file"] + [v.predictions_path for v in variants],
            ["report file"] + [v.report_path for v in variants],
            ["grading helper", c.GRADER_PATH, c.GRADER_PATH],
        ]
    rows += [
        ["submit tool"] + [submit_text(v.submit) for v in variants],
        ["turn countdown"] + ["after every reply" if v.countdown else "none" for v in variants],
        ["scripted follow-ups"] + [
            ", ".join(f"{c} context" for c, f in v.follow_ups.items() if f) or "none" for v in variants
        ],
    ]
    if task == "backdoor":
        rows.append(["sandbox user", "root, files at /", f"{m['realistic'].USER}, files in {m['realistic'].HOME}"])
    else:
        c, r = m["constants"], m["realistic"]
        rows.append(["sandbox user", f"{c.USER}, files in {c.WORKSPACE}", f"{r.USER}, files in {r.PROJECT}"])
    return rows


def task_payload(task: str) -> dict:
    m = load_task(task)
    variants = list(m["variant"].VARIANTS.values())
    scorer = m["scorer"]
    return {
        "name": task,
        "contexts": m["dataset"].CONTEXTS,
        "gradings": m["dataset"].GRADINGS,
        "variants": [
            {
                "name": v.name,
                "prompts": v.prompts,
                "gradings": v.gradings,
                "turn_limits": v.turn_limits,
                "follow_ups": {c: {str(n): msg for n, msg in f.items()} for c, f in v.follow_ups.items()},
                "countdown": v.countdown,
            }
            for v in variants
        ],
        "levers": levers(task, m, variants),
        "judge_criteria": scorer.JUDGE_CRITERIA,
        "judge_templates": {
            "Report judge": scorer.REPORT_JUDGE_TEMPLATE,
            "Report and trajectory judge": scorer.TRAJECTORY_JUDGE_TEMPLATE,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots/prompts.html", help="page to write")
    args = parser.parse_args()

    payload = {
        "tasks": [task_payload(task) for task in TASKS],
        "countdown": {
            "tool_call": TURNS_LEFT.format(turns="N"),
            "no_tool_call": TURNS_LEFT_NO_TOOL_CALL.format(turns="N", submit="submit"),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `</` would end the data script element early, whatever it sits inside.
    data = json.dumps(payload).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.read_text().replace("__DATA__", data))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
