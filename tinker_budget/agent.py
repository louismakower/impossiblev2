"""Run any agent with a budgeted Tinker service alongside it."""

from functools import partial

import anyio
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.util import sandbox, sandbox_service

from tinker_budget.host import SERVICE_NAME, Host


@agent
def with_tinker_budget(inner: Agent, cap_usd: float, use_original_prices: bool = False) -> Agent:
    """Wrap `inner` so its sandbox can call Tinker, capped at `cap_usd`.

    The service polls for requests for as long as the inner agent runs; the
    ledger and every billable event are mirrored into the sample store under
    `tinker_budget`. Calls still in flight when the agent finishes are
    abandoned, and their reservations are never settled, which is harmless
    because the ledger dies with the sample.
    """

    async def execute(state: AgentState) -> AgentState:
        host = Host(cap_usd, use_original_prices)
        done = False
        started = anyio.Event()
        async with anyio.create_task_group() as tg:
            host.tasks = tg
            tg.start_soon(
                partial(sandbox_service, SERVICE_NAME, host.methods(), lambda: done, sandbox(), started=started)
            )
            try:
                await started.wait()  # the client module must exist before the agent can import it
                return await inner(state)
            finally:
                done = True
                tg.cancel_scope.cancel()

    return execute
