"""Per-token prices for every model Tinker serves.

Prices come from the models.json the Tinker docs render their pricing table
from, keyed by exact `tinker_id`: long-context variants like `...:peft:262144`
are separately priced entries. Money is Decimal throughout, and an id that is
not in the table is refused rather than priced by guesswork.
"""

import time
from decimal import Decimal

import httpx

MODELS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
TTL_SECONDS = 3600
METERS = ("prefill", "cached_prefill", "sample", "train")

_models: list[dict] | None = None
_fetched_at = 0.0


class UnknownModel(Exception):
    def __init__(self, tinker_id: str):
        super().__init__(f"{tinker_id!r} is not in Tinker's price list")
        self.tinker_id = tinker_id


async def rates(tinker_id: str, use_original_prices: bool = False) -> dict[str, Decimal]:
    """USD per token for each meter of `tinker_id`.

    With `use_original_prices`, a model on a temporary discount is billed at
    the `original_*` price it will return to.
    """
    for model in await _price_list():
        if model["tinker_id"] == tinker_id:
            return {meter: _price(model, meter, use_original_prices) for meter in METERS}
    raise UnknownModel(tinker_id)


def _price(model: dict, meter: str, use_original_prices: bool) -> Decimal:
    key = f"original_{meter}"
    if not use_original_prices or key not in model:
        key = meter
    return Decimal(model[key].removeprefix("$")) / 1_000_000


async def _price_list() -> list[dict]:
    """The cached list, refreshed once per TTL.

    A failed refresh keeps the last good copy; a failure with nothing cached
    propagates, so nothing gets billed against an unknown price. The failure
    still resets the clock so a dead endpoint costs one timeout per TTL.
    """
    global _models, _fetched_at
    if _models is None or time.monotonic() - _fetched_at > TTL_SECONDS:
        _fetched_at = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(MODELS_URL)
            response.raise_for_status()
            _models = response.json()
        except httpx.HTTPError:
            if _models is None:
                # raise original error with a bare raise
                raise
    return _models
