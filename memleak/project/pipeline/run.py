"""Run a job: pull records from its source, pass each through the steps,
write the results and a manifest describing the run."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .registry import describe
from .source import Record, Source

Step = Callable[[Record], Record]


@dataclass
class Job:
    name: str
    source: Source
    steps: list[Step] = field(default_factory=list)
    limit: int = 100


def run(job: Job, out_dir: str | Path) -> list[Record]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = describe(
        {"name": job.name, "source": job.source, "steps": job.steps, "limit": job.limit}
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    results: list[Record] = []
    for _ in range(job.limit):
        record = job.source.fetch()
        if record is None:
            break
        for step in job.steps:
            record = step(record)
        results.append(record)

    with (out / "results.jsonl").open("w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    return results
