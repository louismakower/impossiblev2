"""Where records come from."""

from collections.abc import Callable, Iterable
from typing import Any

Record = dict[str, Any]


class SourceAPI:
    """One backend. Subclasses hand out records one at a time, `None` when
    there are no more."""

    name: str

    def fetch(self) -> Record | None:
        raise NotImplementedError


class MockSourceAPI(SourceAPI):
    """Canned records for tests: an iterable handed out in order, or a
    callable asked for the next record each time."""

    name = "mock"

    def __init__(self, outputs: Iterable[Record] | Callable[[], Record]):
        self._next = outputs if callable(outputs) else iter(outputs).__next__

    def fetch(self) -> Record | None:
        try:
            return self._next()
        except StopIteration:
            return None


class Source:
    """A backend together with the arguments it was created from, kept so a
    run's manifest can say how to build it again."""

    def __init__(self, api: SourceAPI, args: dict[str, Any]):
        self.api = api
        self.args = args

    def fetch(self) -> Record | None:
        return self.api.fetch()


PROVIDERS: dict[str, type[SourceAPI]] = {MockSourceAPI.name: MockSourceAPI}


def get_source(provider: str, **kwargs: Any) -> Source:
    return Source(PROVIDERS[provider](**kwargs), kwargs)
