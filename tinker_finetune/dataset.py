"""One sample for now: the single prompt in `context.py`.

The data (a Pile of Law slice and LegalBench, built by `prepare_data.py`) is
mounted into the sandbox by `compose.yaml`, so the sample carries no files.
Contexts and submission variants can be crossed here later the way
`test_train/dataset.py` does.
"""

from inspect_ai.dataset import Sample

from context import agent_prompt


def samples() -> list[Sample]:
    return [Sample(id="lora_ranks", input=agent_prompt())]
