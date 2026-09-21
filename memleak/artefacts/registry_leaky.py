"""The manifest writer as it was when the machine went down: an iterator
handed to the mock source reaches `to_jsonable_python`, which lists it until
memory runs out. The shipped `pipeline/registry.py` names iterators instead.
`capture.sh` mounts this over it, with `smoke_leaky.py`, to generate the
incident files.
"""

from typing import Any

from pydantic_core import to_jsonable_python

from .source import Source


def describe(value: Any) -> Any:
    """Turn a job's arguments into something JSON can hold.

    Sources are replaced by their provider and creation arguments, so the
    manifest records how to build them again rather than the live object.
    Anything else JSON can't represent is named rather than dropped.
    """
    if isinstance(value, Source):
        return describe({"provider": value.api.name, "args": value.args})
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return to_jsonable_python(value, fallback=lambda x: getattr(x, "__name__", repr(x)))
