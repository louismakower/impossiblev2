"""The host side of the budgeted Tinker service.

Runs inside the Inspect process, which holds the real `TINKER_API_KEY`. The
sandbox only ever sees opaque client handles and job ids. Every method takes
and returns JSON; `sandbox/tinker_budget.py` is the mirror image that turns
the SDK's types into that JSON and back.

Billable calls are split in two. `submit` estimates the cost, reserves it,
starts the Tinker call in the background and returns a job id at once; `poll`
hands back the result when it is done. Inspect's service loop handles one
batch of requests to completion before looking for more, so a method that
awaited a two-minute training call inline would stall every other request
from that sandbox, including budget checks.
"""

from decimal import Decimal
from typing import Any
from uuid import uuid4

import anyio
import tinker
from inspect_ai.util import store
from tinker import types

from tinker_budget.ledger import BudgetExceeded, Ledger, to_micro, to_usd
from tinker_budget.rates import UnknownModel, rates

SERVICE_NAME = "tinker_rpc"
STORE_KEY = "tinker_budget"

# Checkpoints outlive the sample and cost storage on the shared account, so
# every save gets at most this TTL; an agent may ask for shorter, not longer.
CHECKPOINT_TTL_SECONDS = 2 * 24 * 3600


def _ttl(requested: int | None) -> int:
    return min(requested or CHECKPOINT_TTL_SECONDS, CHECKPOINT_TTL_SECONDS)

# Methods whose Tinker call is free. Logged at zero cost so the eval log shows
# the full call sequence.
FREE = {
    "optim_step",
    "save_state",
    "load_state",
    "load_state_with_optimizer",
    "save_weights_for_sampler",
}


class Job:
    def __init__(self, method: str, reserved: int):
        self.method = method
        self.reserved = reserved
        self.done = False
        self.result: Any = None
        self.error: str | None = None
        self.cost_usd = 0.0


class Host:
    def __init__(self, cap_usd: float, use_original_prices: bool = False):
        self.ledger = Ledger(cap=to_micro(cap_usd))
        self.use_original_prices = use_original_prices
        self.service = tinker.ServiceClient()
        self.training: dict[str, tinker.TrainingClient] = {}
        self.sampling: dict[str, tinker.SamplingClient] = {}
        self.models: dict[str, str] = {}  # handle -> tinker_id
        self.tokenizers: dict[str, Any] = {}
        self.jobs: dict[str, Job] = {}
        self.tasks: anyio.abc.TaskGroup | None = None  # set by the agent wrapper
        self._sync_store()

    def methods(self) -> dict[str, Any]:
        public = [
            self.budget,
            self.create_lora_training_client,
            self.create_training_client_from_state,
            self.create_training_client_from_state_with_optimizer,
            self.create_sampling_client,
            self.training_create_sampling_client,
            self.save_weights_and_get_sampling_client,
            self.get_info,
            self.get_base_model,
            self.encode,
            self.decode,
            self.apply_chat_template,
            self.estimate,
            self.submit,
            self.poll,
        ]
        return {m.__name__: _structured_errors(m) for m in public}

    # -- budget -------------------------------------------------------------

    async def budget(self) -> dict:
        return self.ledger.status()

    def _sync_store(self) -> None:
        store().set(STORE_KEY, self.ledger.snapshot())

    async def _rates(self, handle: str) -> dict[str, Decimal]:
        return await rates(self.models[handle], self.use_original_prices)

    # -- clients ------------------------------------------------------------

    async def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ) -> dict:
        await rates(base_model, self.use_original_prices)  # refuse unknown models before any Tinker call
        client = await self.service.create_lora_training_client_async(
            base_model, rank, seed, train_mlp, train_attn, train_unembed
        )
        return {"handle": self._register_training(client, base_model)}

    async def create_training_client_from_state(self, path: str, base_model: str | None = None) -> dict:
        client = await self.service.create_training_client_from_state_async(path, base_model)
        return {"handle": await self._register_training_from_state(client, base_model)}

    async def create_training_client_from_state_with_optimizer(
        self, path: str, base_model: str | None = None
    ) -> dict:
        client = await self.service.create_training_client_from_state_with_optimizer_async(
            path, base_model
        )
        return {"handle": await self._register_training_from_state(client, base_model)}

    async def _register_training_from_state(
        self, client: tinker.TrainingClient, base_model: str | None
    ) -> str:
        if base_model is None:
            info = await client.get_info_async()
            base_model = info.model_data.model_name
        await rates(base_model, self.use_original_prices)
        return self._register_training(client, base_model)

    def _register_training(self, client: tinker.TrainingClient, model: str) -> str:
        handle = f"training-{uuid4().hex[:8]}"
        self.training[handle] = client
        self.models[handle] = model
        self.ledger.log("create_training_client", model)
        self._sync_store()
        return handle

    async def create_sampling_client(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> dict:
        client = await self.service.create_sampling_client_async(model_path, base_model)
        return {"handle": await self._register_sampling(client)}

    async def training_create_sampling_client(self, handle: str, model_path: str) -> dict:
        client = await self.training[handle].create_sampling_client_async(model_path)
        return {"handle": await self._register_sampling(client)}

    async def save_weights_and_get_sampling_client(self, handle: str, name: str | None = None) -> dict:
        # Not the SDK's own method: that one saves with no expiry.
        training = self.training[handle]
        future = await training.save_weights_for_sampler_async(
            name or f"sampler-{uuid4().hex[:8]}", ttl_seconds=CHECKPOINT_TTL_SECONDS
        )
        saved = await future.result_async()
        self.ledger.log("save_weights_for_sampler", self.models[handle])
        client = await training.create_sampling_client_async(saved.path)
        return {"handle": await self._register_sampling(client)}

    async def _register_sampling(self, client: tinker.SamplingClient) -> str:
        model = await client.get_base_model_async()
        await rates(model, self.use_original_prices)
        handle = f"sampling-{uuid4().hex[:8]}"
        self.sampling[handle] = client
        self.models[handle] = model
        self.ledger.log("create_sampling_client", model)
        self._sync_store()
        return handle

    async def get_info(self, handle: str) -> dict:
        info = await self.training[handle].get_info_async()
        return info.model_dump(mode="json")

    async def get_base_model(self, handle: str) -> str:
        return self.models[handle]

    # -- tokenizer ----------------------------------------------------------
    # Tokenising is local CPU work on a tokenizer downloaded from Hugging Face:
    # no Tinker request, no cost. The sandbox has no network, so it happens here.

    async def _tokenizer(self, handle: str) -> Any:
        if handle not in self.tokenizers:
            client = self.training.get(handle) or self.sampling[handle]
            self.tokenizers[handle] = await anyio.to_thread.run_sync(client.get_tokenizer)
        return self.tokenizers[handle]

    async def encode(self, handle: str, text: str, add_special_tokens: bool = False) -> list[int]:
        tokenizer = await self._tokenizer(handle)
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    async def decode(self, handle: str, tokens: list[int], skip_special_tokens: bool = False) -> str:
        tokenizer = await self._tokenizer(handle)
        return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    async def apply_chat_template(
        self, handle: str, messages: list[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        tokenizer = await self._tokenizer(handle)
        # Render to text then encode: what tokenize=True returns varies across
        # transformers versions (a list, or a BatchEncoding that is not JSON).
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        return tokenizer.encode(text, add_special_tokens=False)

    # -- billable calls -----------------------------------------------------

    # `op` rather than `method`: the generated client's `call_<service>(method, **params)`
    # already uses that name for the service method.
    async def estimate(self, op: str, handle: str, **params: Any) -> dict:
        """Worst-case cost of a call without making it, so the agent can plan."""
        micro, tokens = await self._estimate(op, handle, params)
        return {"estimated_usd": to_usd(micro), "tokens": tokens}

    async def _estimate(self, method: str, handle: str, params: dict) -> tuple[int, dict[str, int]]:
        if method in FREE:
            return 0, {}
        r = await self._rates(handle)
        if method == "sample":
            prompt = types.ModelInput.model_validate(params["prompt"]).length
            n = params["num_samples"]
            max_tokens = params["sampling_params"].get("max_tokens")
            if max_tokens is None:
                raise ValueError("sampling_params.max_tokens is required so the cost can be bounded")
            # The prompt is prefilled once; the other num_samples - 1 copies bill as cache hits.
            cost = (
                prompt * r["prefill"]
                + prompt * (n - 1) * r["cached_prefill"]
                + max_tokens * n * r["sample"]
            )
            return to_micro(cost), {"prompt": prompt, "max_generated": max_tokens * n}
        if method == "compute_logprobs":
            prompt = types.ModelInput.model_validate(params["prompt"]).length
            return to_micro(prompt * r["prefill"]), {"prompt": prompt}
        if method in ("forward", "forward_backward"):
            tokens = sum(types.ModelInput.model_validate(d["model_input"]).length for d in params["data"])
            meter = "prefill" if method == "forward" else "train"
            return to_micro(tokens * r[meter]), {"input": tokens}
        raise ValueError(f"unknown method {method!r}")

    async def submit(self, op: str, handle: str, **params: Any) -> dict:
        reserved, tokens = await self._estimate(op, handle, params)
        self.ledger.reserve(reserved)  # raises BudgetExceeded before any Tinker call
        self._sync_store()
        job = Job(op, reserved)
        job_id = uuid4().hex
        self.jobs[job_id] = job
        assert self.tasks is not None, "Host.tasks must be set before submitting jobs"
        self.tasks.start_soon(self._run, job, handle, params, tokens)
        return {"job": job_id, "estimated_usd": to_usd(reserved)}

    async def poll(self, job: str) -> dict:
        j = self.jobs[job]
        if not j.done:
            return {"done": False}
        del self.jobs[job]
        return {"done": True, "result": j.result, "error": j.error, "cost_usd": j.cost_usd}

    async def _run(self, job: Job, handle: str, params: dict, tokens: dict[str, int]) -> None:
        """Make the call, then settle to what it actually cost.

        A failed call settles to 0 on the assumption Tinker does not bill
        requests it rejected. Anything else unexpected settles to the estimate:
        overcharging the agent is the safe side.
        """
        actual = job.reserved
        try:
            job.result, actual = await self._call(job.method, handle, params, tokens)
        except Exception as e:
            job.error = str(e)
            actual = 0
        finally:
            self.ledger.settle(job.reserved, actual, job.method, self.models[handle], tokens)
            job.cost_usd = to_usd(actual)
            job.done = True
            self._sync_store()

    async def _call(self, method: str, handle: str, params: dict, tokens: dict[str, int]) -> tuple[Any, int]:
        """Returns (json result, actual cost in micro-dollars)."""
        if method == "sample":
            client = self.sampling[handle]
            r = await self._rates(handle)
            response = await client.sample_async(
                prompt=types.ModelInput.model_validate(params["prompt"]),
                num_samples=params["num_samples"],
                sampling_params=types.SamplingParams.model_validate(params["sampling_params"]),
                include_prompt_logprobs=params.get("include_prompt_logprobs", False),
                topk_prompt_logprobs=params.get("topk_prompt_logprobs", 0),
            )
            prompt, n = tokens["prompt"], params["num_samples"]
            hits = response.prompt_cache_hit_tokens
            generated = sum(len(s.tokens) for s in response.sequences)
            tokens["generated"] = generated
            tokens["cached_prefill"] = hits + prompt * (n - 1)
            cost = (
                (prompt - hits) * r["prefill"]
                + tokens["cached_prefill"] * r["cached_prefill"]
                + generated * r["sample"]
            )
            return _sample_response_json(response), to_micro(cost)

        if method == "compute_logprobs":
            client = self.sampling[handle]
            r = await self._rates(handle)
            logprobs = await client.compute_logprobs_async(types.ModelInput.model_validate(params["prompt"]))
            return logprobs, to_micro(tokens["prompt"] * r["prefill"])

        client = self.training[handle]
        if method in ("forward", "forward_backward"):
            data = [_datum(d) for d in params["data"]]
            run = client.forward_async if method == "forward" else client.forward_backward_async
            future = await run(data, params["loss_fn"], params.get("loss_fn_config"))
            output = await future.result_async()
            r = await self._rates(handle)
            meter = "prefill" if method == "forward" else "train"
            return _fwd_bwd_output_json(output), to_micro(tokens["input"] * r[meter])

        if method == "optim_step":
            future = await client.optim_step_async(types.AdamParams.model_validate(params["adam_params"]))
        elif method == "save_state":
            future = await client.save_state_async(
                params["name"], _ttl(params.get("ttl_seconds")), params.get("overwrite", False)
            )
        elif method == "load_state":
            future = await client.load_state_async(params["path"])
        elif method == "load_state_with_optimizer":
            future = await client.load_state_with_optimizer_async(params["path"])
        elif method == "save_weights_for_sampler":
            future = await client.save_weights_for_sampler_async(params["name"], _ttl(params.get("ttl_seconds")))
        else:
            raise ValueError(f"unknown method {method!r}")
        response = await future.result_async()
        return response.model_dump(mode="json"), 0


# -- JSON conversions ---------------------------------------------------------
# Inputs the agent builds are pydantic and validate straight from JSON. The
# SDK's outputs and Datum are frozen dataclasses over numpy, so those are
# converted by hand here and rebuilt by hand in the sandbox module.


def _datum(d: dict) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.model_validate(d["model_input"]),
        loss_fn_inputs={k: types.TensorData(**v) for k, v in d["loss_fn_inputs"].items()},
    )


def _tensor_json(td: types.TensorData) -> dict:
    return {
        "data": td.data,
        "dtype": td.dtype,
        "shape": td.shape,
        "sparse_crow_indices": td.sparse_crow_indices,
        "sparse_col_indices": td.sparse_col_indices,
    }


def _sample_response_json(response: types.SampleResponse) -> dict:
    return {
        "sequences": [
            {
                "stop_reason": s.stop_reason,
                "sequence_id": s.sequence_id,
                "tokens": s.tokens,
                "logprobs": s.logprobs,
            }
            for s in response.sequences
        ],
        "prompt_logprobs": response.prompt_logprobs,
        "topk_prompt_logprobs": response.topk_prompt_logprobs,
        "prompt_cache_hit_tokens": response.prompt_cache_hit_tokens,
    }


def _fwd_bwd_output_json(output: types.ForwardBackwardOutput) -> dict:
    return {
        "loss_fn_output_type": output.loss_fn_output_type,
        "loss_fn_outputs": [{k: _tensor_json(v) for k, v in out.items()} for out in output.loss_fn_outputs],
        "metrics": output.metrics,
    }


def _structured_errors(method):
    """Return budget, model and argument errors as data so the sandbox can re-raise them by type.

    Every other exception is left to Inspect, which delivers it as a plain
    error message.
    """

    async def wrapped(**params):
        try:
            return await method(**params)
        except (BudgetExceeded, UnknownModel, ValueError) as e:
            return {"error_type": type(e).__name__, "message": str(e)}

    wrapped.__name__ = method.__name__
    return wrapped
