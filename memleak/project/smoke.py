"""Smoke test: run a small job end to end against the mock source.

    python smoke.py
"""

import tempfile

from pipeline.run import Job, run
from pipeline.source import get_source


def double(record):
    return {**record, "value": record["value"] * 2}


def record():
    return {"value": 21}


def main():
    source = get_source("mock", outputs=record)
    job = Job(name="smoke", source=source, steps=[double], limit=1000)
    with tempfile.TemporaryDirectory() as out:
        results = run(job, out)
    assert len(results) == 1000
    assert all(r["value"] == 42 for r in results)
    print(f"ok: {len(results)} records")


if __name__ == "__main__":
    main()
