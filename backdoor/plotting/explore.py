"""Build an interactive version of the honesty plot.

Same grouped bars as `analyse.py` writes to `honesty.png`, but as an HTML page:
clicking a segment lists the samples behind it, each with the evidence it was
graded on (the conversation JSON, the tool call that returned the phrase, or
the report), the scorer's verdict, and every scanner's verdict and explanation.

Usage:
    uv run backdoor/plotting/explore.py logs/backdoor-grid/*.eval
    uv run backdoor/plotting/explore.py logs/backdoor-grid/*.eval --out plots/honesty.html
"""

import argparse
import json
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_scout import scan_list, scan_results_df

# The eval's own modules live one directory up, and are imported flat there too.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyse import CONTEXTS, HONESTY_LABELS, SUBMISSIONS, parse_value
from common import CONVERSATION_PATH, REPORT_PATH

TEMPLATE = Path(__file__).parent / "explorer.html"

# The tool call id the tool_reader scorer names when it finds the phrase.
TOOL_CALL_ID = re.compile(r"at tool call (\S+)")


def evidence(sample, submission: str, score) -> list[dict]:
    """What the sample was graded on, as blocks to show under the chart."""
    if submission == "json":
        return conversation_file(sample)
    if submission == "tool":
        return target_tool_call(sample, score)
    return [{"title": f"report ({REPORT_PATH})", "language": "markdown", "text": score.answer}]


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
        return []

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
                    {"title": CONVERSATION_PATH, "language": "json",
                     "text": json.dumps(parsed, indent=2)},
                    {"title": "the command that wrote it", "language": "bash", "text": command},
                ]
    return [{"title": "the command that wrote it", "language": "bash", "text": command}]


def target_tool_call(sample, score) -> list[dict]:
    """The `llm` call the tool_reader scorer credited, request and response."""
    match = TOOL_CALL_ID.search(score.explanation or "")
    if not match:
        return []
    call_id = match.group(1)

    blocks = []
    for message in sample.messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.id == call_id:
                blocks.append({
                    "title": "the request that elicited it",
                    "language": "json",
                    "text": json.dumps(call.arguments, indent=2),
                })
        if getattr(message, "tool_call_id", None) == call_id:
            blocks.append({"title": "the response", "language": "json", "text": message.text})
    return blocks


def scan_results(scan_dirs: list[str]) -> dict[str, dict]:
    """Latest verdict and explanation per (sample, scanner)."""
    results: dict[str, dict] = {}
    for scans in scan_dirs:
        if not Path(scans).is_dir():
            continue
        for status in sorted(scan_list(scans), key=lambda s: s.spec.timestamp):
            for name, df in scan_results_df(status.location).scanners.items():
                for _, row in df.iterrows():
                    value = parse_value(row["value"])
                    answer = row.get("answer")
                    # Single-choice scanners put the letter in `value` and the
                    # text in `answer`; multi-label ones do the opposite.
                    if isinstance(value, str) and len(value) <= 2 and isinstance(answer, str):
                        value = answer
                    results.setdefault(str(row["transcript_id"]), {})[name] = {
                        "value": value,
                        "explanation": row.get("explanation"),
                    }
    return results


def samples(logs: list[str], scans: dict[str, dict]) -> list[dict]:
    rows = []
    for path in logs:
        log = read_eval_log(path)
        model = log.eval.model.split("/")[-1]
        for sample in log.samples or []:
            metadata = sample.metadata or {}
            submission = metadata.get("submission", "unknown")
            name, score = next(iter(sample.scores.items()))
            rows.append({
                "id": sample.uuid or f"{sample.id}-{sample.epoch}",
                "model": model,
                "submission": submission,
                "context": metadata.get("context", "unknown"),
                "epoch": sample.epoch,
                "score": {
                    "scorer": name,
                    "value": str(score.value),
                    "answer": score.answer,
                    "explanation": score.explanation,
                },
                "scanners": scans.get(sample.uuid, {}),
                "evidence": [b for b in evidence(sample, submission, score) if b.get("text")],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="eval log files to include")
    parser.add_argument("--scans", action="append", default=None,
                        help="scan results directory (repeatable, defaults to ./scans)")
    parser.add_argument("--out", default="plots/honesty.html", help="page to write")
    args = parser.parse_args()

    rows = samples(args.logs, scan_results(args.scans or ["scans"]))
    payload = {
        # Ordered honest-first, like the plot; the page holds the colours.
        "bands": [{"key": key, "label": label} for key, label in HONESTY_LABELS.items()],
        "submissions": SUBMISSIONS,
        "contexts": CONTEXTS,
        "samples": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `</` would end the data script element early, whatever it sits inside.
    data = json.dumps(payload).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("__DATA__", data))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(rows)} samples)")


if __name__ == "__main__":
    main()
