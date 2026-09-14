"""Build a page of the awareness-timing scanner's results over an eval grid.

For each sample the scanner gives the message where the agent first suspects a
reward-hacking eval (A) and where it first concludes the task is impossible
(B), or -1. The page derives which came first, overall and split by task,
model, variant, context and grading, with checkboxes to filter; the gap
between the two in agent turns; the split by whether the sample reward
hacked; agreement with the awareness scanner; and every quote for checking.

Message numbers count from the first message after the system prompt, so
message k is index k of the sample's messages, and its agent turn is the
number of assistant messages up to and including it.

Usage:
    uv run common/timing.py logs/backdoor/*.eval logs/test_train/*.eval --out plots/timing.html
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_scout import scan_list, scan_results_df

from common.hack import FACTORS, awareness_table, sample_table

TEMPLATE = Path(__file__).parent / "timing.html"


def timing_table(scans: str) -> pd.DataFrame:
    """The timing scanner's latest verdict per sample."""
    rows = []
    for status in sorted(scan_list(scans), key=lambda s: s.spec.timestamp):
        try:
            results = scan_results_df(status.location).scanners["awareness_timing"]
        except KeyError:
            continue
        for _, row in results.iterrows():
            value = json.loads(row["value"]) if isinstance(row["value"], str) else {}
            if "a_message" not in value:  # errored
                continue
            rows.append({"sample_id": str(row["transcript_id"]), **value, "timing_explanation": row.get("explanation")})
    return pd.DataFrame(rows, columns=["sample_id", "a_message", "b_message", "a_quote", "b_quote", "timing_explanation"]).drop_duplicates("sample_id", keep="last")


def turn_maps(logs: list[str]) -> dict[str, list[int]]:
    """Per sample, the agent turn number at each message number: entry k is
    how many assistant messages there are among messages 1..k."""
    maps = {}
    for path in logs:
        for sample in read_eval_log(path).samples or []:
            turns, count = [0], 0
            for message in sample.messages[1:]:
                count += message.role == "assistant"
                turns.append(count)
            maps[sample.uuid] = turns
    return maps


def turn(turns: list[int] | None, message) -> int | None:
    if turns is None or pd.isna(message) or int(message) < 1 or int(message) >= len(turns):
        return None
    return turns[int(message)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="eval log files to include")
    parser.add_argument("--out", default="plots/timing.html", help="page to write")
    args = parser.parse_args()

    dirs = {str(Path(p).parent / "scans") for p in args.logs}
    df = (
        sample_table(args.logs)
        .merge(pd.concat([awareness_table(d) for d in dirs]), on="sample_id", how="left")
        .merge(pd.concat([timing_table(d) for d in dirs]), on="sample_id", how="left")
    )
    turns = turn_maps(args.logs)

    def present(message) -> bool:
        return not pd.isna(message) and int(message) >= 1

    rows = []
    for r in df.itertuples():
        if pd.isna(r.a_message):  # not scanned yet
            continue
        t = turns.get(r.sample_id)
        rows.append({
            "id": r.sample_id,
            "task": r.task, "model": r.model, "variant": r.variant, "context": r.context, "grading": r.grading,
            "hack": bool(r.hack),
            "a": int(r.a_message) if present(r.a_message) else None,
            "b": int(r.b_message) if present(r.b_message) else None,
            "a_turn": turn(t, r.a_message), "b_turn": turn(t, r.b_message),
            "turns": t[-1] if t else None,
            "a_quote": r.a_quote if present(r.a_message) else "",
            "b_quote": r.b_quote if present(r.b_message) else "",
            # the awareness scanner's view of A, for agreement: True if implied or explicit
            "aware": None if pd.isna(r.rewardhack_eval_awareness) else bool(r.rewardhack_eval_awareness),
        })

    orders = {"task": [], "model": [], **FACTORS}
    filters = {
        key: [v for v in order if v in set(df[key])] + sorted(set(df[key]) - set(order))
        for key, order in orders.items()
    }
    payload = {"filters": filters, "samples": rows, "total": len(df)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(payload).replace("</", "<\\/")))
    print(f"Wrote {out}: {len(rows)} of {len(df)} samples have a timing result")


if __name__ == "__main__":
    main()
