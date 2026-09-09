"""One agent's spend against its cap, in integer micro-dollars.

Every billable call reserves its worst case before it is made and settles to
the actual cost after. The estimate is the cap: Tinker's own billing lags by
hours, so nothing here is reconciled against it. Reserve and settle never
await, which is all the atomicity a single asyncio process needs.
"""

from dataclasses import asdict, dataclass, field
from decimal import ROUND_CEILING, Decimal
from time import time

MICRO = 1_000_000


def to_micro(usd: Decimal | float) -> int:
    """Round up, so estimates never under-reserve.

    Floats go through `str` first: `Decimal(0.1)` is 0.1000000000000000055...,
    which would round up to a micro-dollar that was never spent.
    """
    if isinstance(usd, float):
        usd = Decimal(str(usd))
    return int((usd * MICRO).to_integral_value(ROUND_CEILING))


def to_usd(micro: int) -> float:
    return micro / MICRO


class BudgetExceeded(Exception):
    def __init__(self, requested_micro: int, remaining_micro: int):
        super().__init__(
            f"this call would cost up to ${to_usd(requested_micro):.4f} "
            f"but only ${to_usd(remaining_micro):.4f} of the budget remains"
        )
        self.requested_usd = to_usd(requested_micro)
        self.remaining_usd = to_usd(remaining_micro)


@dataclass
class Event:
    """One settled call. Free calls are logged too, so the sequence is visible."""

    ts: float
    method: str
    model: str | None
    tokens: dict[str, int]
    estimated_usd: float
    cost_usd: float


@dataclass
class Ledger:
    cap: int
    spent: int = 0
    reserved: int = 0
    events: list[Event] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.cap - self.spent - self.reserved

    def reserve(self, amount: int) -> None:
        if amount > self.remaining:
            raise BudgetExceeded(amount, self.remaining)
        self.reserved += amount

    def settle(self, reserved: int, actual: int, method: str, model: str | None, tokens: dict[str, int]) -> None:
        """Release `reserved` and charge `actual`; 0 releases a failed call.

        `actual` may exceed `reserved` when the estimate was low. Spend then
        passes the cap and `remaining` goes negative, which refuses every
        later reserve; that is the accepted drift, not a bug.
        """
        self.reserved -= reserved
        self.spent += actual
        self.events.append(
            Event(time(), method, model, tokens, to_usd(reserved), to_usd(actual))
        )

    def log(self, method: str, model: str | None) -> None:
        """Record a free call that never went through reserve/settle."""
        self.events.append(Event(time(), method, model, {}, 0.0, 0.0))

    def status(self) -> dict[str, float]:
        return {
            "cap_usd": to_usd(self.cap),
            "spent_usd": to_usd(self.spent),
            "reserved_usd": to_usd(self.reserved),
            "remaining_usd": to_usd(self.remaining),
        }

    def snapshot(self) -> dict:
        return self.status() | {"events": [asdict(e) for e in self.events]}
