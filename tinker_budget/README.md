Tinker for sandboxed agents, with a per-sample dollar cap.

The agent's code runs in the sandbox and imports `tinker_budget` (see
`sandbox/tinker_budget.py` for the API, which mirrors the Tinker SDK). Calls
cross into the Inspect process over a sandbox service, where `host.py` holds
the real key, prices each call from `rates.py`, and charges `ledger.py`. Spend
and every call land in the sample store under `tinker_budget`.

build the sandbox image (task images can `FROM tinker_budget:latest`)
`docker build -t tinker_budget:latest tinker_budget/`

wrap any agent
```python
from tinker_budget import with_tinker_budget
solver=with_tinker_budget(react(...), cap_usd=5.0)
```

`TINKER_API_KEY` is read from the root `.env`; run `inspect eval` from the repo root.

end-to-end check against real Tinker (spends well under a cent)
`inspect eval tinker_budget/smoke/smoke.py --model mockllm/model`
