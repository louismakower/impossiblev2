"""Interactive scanner explorer for the backdoor grid, see `common.explore`.

Usage:
    uv run backdoor/plotting/explore.py logs/backdoor-grid/*.eval
    uv run backdoor/plotting/explore.py logs/backdoor-grid/*.eval --out plots/backdoor.html
"""

import json
import re
import sys
from pathlib import Path

# The eval's own modules live one directory up, and are imported flat there too.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyse import CONTEXTS, SUBMISSIONS
from constants import CONVERSATION_PATH, REPORT_PATH

from common.explore import main, report_block

# The tool call id the tool_reader scorer names when it finds the phrase.
TOOL_CALL_ID = re.compile(r"at tool call (\S+)")


def evidence(sample, submission: str, score) -> list[dict]:
    """What the sample was graded on, as blocks to show under the chart."""
    if submission == "json":
        return conversation_file(sample)
    if submission == "tool":
        return target_tool_call(sample, score)
    return report_block(score, REPORT_PATH)


def conversation_file(sample) -> list[dict]:
    """The conversation file, recovered from the bash call that wrote it.

    Nothing in the log holds the file itself — the scorer reads it from the
    sandbox and keeps only the final assistant message — so the closest thing
    is the last command that mentions the path.
    """
    written = [
        call.arguments.get("cmd") or call.arguments.get("command") or json.dumps(call.arguments)
        for message in sample.messages
        for call in getattr(message, "tool_calls", None) or []
        if call.function == "bash" and CONVERSATION_PATH in json.dumps(call.arguments)
    ]
    if not written:
        return [{"title": "no graded artifact",
                 "text": "The agent never wrote the file this submission is graded on."}]

    command = written[-1]
    # A heredoc'd file is the common shape, so show the JSON itself when the
    # command contains something that parses, and the raw command otherwise.
    for start in (m.start() for m in re.finditer(r"\{", command)):
        for end in (m.start() for m in re.finditer(r"\}", command[start:])):
            candidate = command[start : start + end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "responses" in parsed:
                return [
                    {"title": CONVERSATION_PATH, "text": json.dumps(parsed, indent=2)},
                    {"title": "the command that wrote it", "text": command},
                ]
    return [{"title": "the command that wrote it", "text": command}]


def target_tool_call(sample, score) -> list[dict]:
    """The `llm` call the tool_reader scorer credited, request and response."""
    match = TOOL_CALL_ID.search(score.explanation or "")
    if not match:
        return [{"title": "no graded artifact",
                 "text": "The scorer credited no tool call, so there is no successful call to show. "
                         "The scorer blocks below say what it found."}]
    call_id = match.group(1)

    blocks = []
    for message in sample.messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.id == call_id:
                blocks.append({
                    "title": "the request that elicited it",
                    "text": json.dumps(call.arguments, indent=2),
                })
        if getattr(message, "tool_call_id", None) == call_id:
            blocks.append({"title": "the response", "text": message.text})
    return blocks


if __name__ == "__main__":
    main(SUBMISSIONS, CONTEXTS, evidence, out="plots/backdoor.html")
