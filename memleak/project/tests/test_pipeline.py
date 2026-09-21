import json

from pipeline.registry import describe
from pipeline.run import Job, run
from pipeline.source import get_source


def add_one(record):
    return {**record, "n": record["n"] + 1}


def test_list_source_runs_out():
    source = get_source("mock", outputs=[{"n": 1}, {"n": 2}])
    assert source.fetch() == {"n": 1}
    assert source.fetch() == {"n": 2}
    assert source.fetch() is None


def test_run_writes_results_and_manifest(tmp_path):
    job = Job(name="t", source=get_source("mock", outputs=[{"n": 1}, {"n": 2}]), steps=[add_one])
    results = run(job, tmp_path)
    assert results == [{"n": 2}, {"n": 3}]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source"] == {"provider": "mock", "args": {"outputs": [{"n": 1}, {"n": 2}]}}
    assert manifest["steps"] == ["add_one"]


def test_describe_names_callables():
    assert describe({"f": add_one}) == {"f": "add_one"}
