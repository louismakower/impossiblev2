"""Build a page comparing awareness between samples that reward hacked and
samples that didn't.

Two charts, one per awareness field. Each has two 100% stacked bars, the
samples that hacked (passed both the deterministic scorer and the report
judge) and the samples that didn't, split by the scanner's level. Checkboxes
filter by task, model, variant, context and grading; the bars recompute in
the page. Logs from both tasks can go in together. Scan results are read from
the `scans` directory beside each log.

Usage:
    uv run common/awareness.py logs/backdoor/*.eval logs/test_train/*.eval --out plots/awareness.html
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common.hack import AWARENESS, FACTORS, awareness_table, sample_table

TEMPLATE = Path(__file__).parent / "awareness.html"


def level(hit, explicit) -> str | None:
    """The scanner's level back from the two booleans `awareness_table` keeps."""
    if pd.isna(hit):
        return None
    return "explicit" if explicit else "implied" if hit else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="eval log files to include")
    parser.add_argument("--out", default="plots/awareness.html", help="page to write")
    args = parser.parse_args()

    df = sample_table(args.logs)
    scans = pd.concat(
        [awareness_table(str(Path(d) / "scans")) for d in {Path(p).parent for p in args.logs}]
    )
    df = df.merge(scans, on="sample_id", how="left")

    rows = [
        {
            "task": r.task,
            "model": r.model,
            "variant": r.variant,
            "context": r.context,
            "grading": r.grading,
            "hack": bool(r.hack),
            **{f: level(getattr(r, f), getattr(r, f"{f}_explicit")) for f in AWARENESS},
        }
        for r in df.itertuples()
    ]

    # filter groups in display order, with anything unrecognised appended
    orders = {"task": [], "model": [], **FACTORS}
    filters = {
        key: [v for v in order if v in set(df[key])] + sorted(set(df[key]) - set(order))
        for key, order in orders.items()
    }

    payload = {"filters": filters, "fields": AWARENESS, "samples": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `</` would end the data script element early
    data = json.dumps(payload).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("__DATA__", data))
    scanned = sum(r["eval_awareness"] is not None for r in rows)
    print(f"Wrote {out}: {len(rows)} samples, {scanned} scanned, {sum(r['hack'] for r in rows)} hacks")


if __name__ == "__main__":
    main()
