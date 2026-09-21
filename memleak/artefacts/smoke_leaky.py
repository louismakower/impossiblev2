"""The smoke test as it was when it took the machine down: the mock source is
given an infinite iterator, and the manifest writer of the time
(`registry_leaky.py`) hands it to `to_jsonable_python`, which lists it until
memory runs out. `capture.sh` mounts both over the shipped project to
generate the incident files. Never run it outside a memory-capped container
or `ulimit -v`.
"""

import itertools
import tempfile

from pipeline.run import Job, run
from pipeline.source import get_source


def double(record):
    return {**record, "value": record["value"] * 2}


def main():
    source = get_source("mock", outputs=itertools.repeat({"value": 21}))
    job = Job(name="smoke", source=source, steps=[double], limit=1000)
    with tempfile.TemporaryDirectory() as out:
        results = run(job, out)
    assert len(results) == 1000
    assert all(r["value"] == 42 for r in results)
    print(f"ok: {len(results)} records")


if __name__ == "__main__":
    main()
