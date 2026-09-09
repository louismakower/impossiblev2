"""Interactive scanner explorer for the backdoor grid, see `common.explore`.

Usage:
    uv run backdoor/plotting/explore.py logs/backdoor/*.eval
    uv run backdoor/plotting/explore.py logs/backdoor/*.eval --out plots/backdoor.html
"""

import json
import re
import sys
from pathlib import Path

# The eval's own modules live one directory up, and are imported flat there too.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyse import CONTEXTS, GRADINGS
from constants import CONVERSATION_PATH, REPORT_PATH

from common.explore import main, report_block


def evidence(sample) -> list[dict]:
    """Both graded artifacts, whatever the agent was told: the conversation
    file and the report the judges read."""
    return conversation_file(sample) + report_block(sample, REPORT_PATH)


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
        return [{"title": "no conversation file",
                 "text": f"The agent never wrote {CONVERSATION_PATH}."}]

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


if __name__ == "__main__":
    main(GRADINGS, CONTEXTS, evidence, out="plots/backdoor.html")
