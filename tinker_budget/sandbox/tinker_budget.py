"""Tinker, with a spending cap.

Use this exactly like the `tinker` SDK: the classes and methods below take
the same arguments and return the same types. The differences are that
`SamplingParams.max_tokens` is required, custom loss functions are not
supported, and every call is charged against a fixed budget.

    import tinker_budget as tinker
    from tinker import types

    svc = tinker.ServiceClient()
    training = svc.create_lora_training_client(base_model="Qwen/Qwen3-8B")
    tok = training.get_tokenizer()

    future = training.forward_backward(data, "cross_entropy")
    print(future.estimated_usd)          # reserved before the call was made
    out = future.result()
    print(future.cost_usd)               # what it actually cost

    print(tinker.budget())               # cap, spent, reserved, remaining
    print(training.estimate_forward_backward(data))   # cost a call without making it

A call that would take the budget past its cap raises `BudgetExceeded`
before anything is sent. A model not on Tinker's price list raises
`UnknownModel`.

Requires the real `tinker` package for its types; it never needs an API key.
"""

import importlib.util
import time
from asyncio import sleep as async_sleep
from typing import Any

import numpy as np
from tinker import types

SERVICE_NAME = "tinker_rpc"
POLL_SECONDS = 0.5


class BudgetExceeded(Exception):
    pass


class UnknownModel(Exception):
    pass


_ERRORS = {"BudgetExceeded": BudgetExceeded, "UnknownModel": UnknownModel, "ValueError": ValueError}


def _load_rpc():
    path = f"/var/tmp/sandbox-services/{SERVICE_NAME}/{SERVICE_NAME}.py"
    spec = importlib.util.spec_from_file_location(f"_{SERVICE_NAME}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rpc = _load_rpc()


def _check(response: Any) -> Any:
    if isinstance(response, dict) and "error_type" in response:
        raise _ERRORS[response["error_type"]](response["message"])
    return response


def _call(method: str, **params: Any) -> Any:
    return _check(getattr(_rpc, f"call_{SERVICE_NAME}")(method, **params))


async def _call_async(method: str, **params: Any) -> Any:
    return _check(await getattr(_rpc, f"call_{SERVICE_NAME}_async")(method, **params))


def budget() -> dict[str, float]:
    """cap_usd, spent_usd, reserved_usd and remaining_usd."""
    return _call("budget")


# -- futures ------------------------------------------------------------------


class Future:
    """A submitted call. `result()` blocks until it finishes; `await` also works.

    `estimated_usd` is known from the start. `cost_usd` is set once the call
    completes, and the difference has already been returned to the budget.
    """

    def __init__(self, submitted: dict, decode):
        self._job = submitted["job"]
        self._decode = decode
        self._response: dict | None = None
        self.estimated_usd: float = submitted["estimated_usd"]
        self.cost_usd: float | None = None

    def result(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._response is None:
            self._take(_call("poll", job=self._job))
            if self._response is None:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError()
                time.sleep(POLL_SECONDS)
        return self._value()

    async def result_async(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._response is None:
            self._take(await _call_async("poll", job=self._job))
            if self._response is None:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError()
                await async_sleep(POLL_SECONDS)
        return self._value()

    def __await__(self):
        return self.result_async().__await__()

    def _take(self, polled: dict) -> None:
        if polled["done"]:
            self._response = polled
            self.cost_usd = polled["cost_usd"]

    def _value(self):
        if self._response["error"] is not None:
            raise RuntimeError(self._response["error"])
        return self._decode(self._response["result"])


def _submit(op: str, handle: str, decode, **params: Any) -> Future:
    return Future(_call("submit", op=op, handle=handle, **params), decode)


async def _submit_async(op: str, handle: str, decode, **params: Any) -> Future:
    return Future(await _call_async("submit", op=op, handle=handle, **params), decode)


# -- clients ------------------------------------------------------------------


class ServiceClient:
    def __init__(self, **_: Any):
        pass

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict | None = None,
    ) -> "TrainingClient":
        created = _call(
            "create_lora_training_client",
            base_model=base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
        return TrainingClient(created["handle"])

    async def create_lora_training_client_async(self, *args, **kwargs) -> "TrainingClient":
        return self.create_lora_training_client(*args, **kwargs)

    def create_training_client_from_state(
        self, path: str, base_model: str | None = None, **_: Any
    ) -> "TrainingClient":
        created = _call("create_training_client_from_state", path=path, base_model=base_model)
        return TrainingClient(created["handle"])

    async def create_training_client_from_state_async(self, *args, **kwargs) -> "TrainingClient":
        return self.create_training_client_from_state(*args, **kwargs)

    def create_training_client_from_state_with_optimizer(
        self, path: str, base_model: str | None = None, **_: Any
    ) -> "TrainingClient":
        created = _call(
            "create_training_client_from_state_with_optimizer", path=path, base_model=base_model
        )
        return TrainingClient(created["handle"])

    async def create_training_client_from_state_with_optimizer_async(self, *args, **kwargs):
        return self.create_training_client_from_state_with_optimizer(*args, **kwargs)

    def create_sampling_client(
        self, model_path: str | None = None, base_model: str | None = None, **_: Any
    ) -> "SamplingClient":
        created = _call("create_sampling_client", model_path=model_path, base_model=base_model)
        return SamplingClient(created["handle"])

    async def create_sampling_client_async(self, *args, **kwargs) -> "SamplingClient":
        return self.create_sampling_client(*args, **kwargs)


class Tokenizer:
    """The model's tokenizer, run on the host. Free."""

    def __init__(self, handle: str):
        self._handle = handle

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return _call("encode", handle=self._handle, text=text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return _call(
            "decode", handle=self._handle, tokens=list(tokens), skip_special_tokens=skip_special_tokens
        )

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        return _call(
            "apply_chat_template",
            handle=self._handle,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )


class TrainingClient:
    def __init__(self, handle: str):
        self._handle = handle

    def get_info(self) -> types.GetInfoResponse:
        return types.GetInfoResponse.model_validate(_call("get_info", handle=self._handle))

    async def get_info_async(self) -> types.GetInfoResponse:
        return self.get_info()

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer(self._handle)

    # training

    def forward_backward(self, data, loss_fn: str, loss_fn_config: dict | None = None) -> Future:
        return _submit(
            "forward_backward", self._handle, _fwd_bwd_output,
            data=[_datum_json(d) for d in data], loss_fn=loss_fn, loss_fn_config=loss_fn_config,
        )

    async def forward_backward_async(self, data, loss_fn: str, loss_fn_config: dict | None = None) -> Future:
        return await _submit_async(
            "forward_backward", self._handle, _fwd_bwd_output,
            data=[_datum_json(d) for d in data], loss_fn=loss_fn, loss_fn_config=loss_fn_config,
        )

    def forward(self, data, loss_fn: str, loss_fn_config: dict | None = None) -> Future:
        return _submit(
            "forward", self._handle, _fwd_bwd_output,
            data=[_datum_json(d) for d in data], loss_fn=loss_fn, loss_fn_config=loss_fn_config,
        )

    async def forward_async(self, data, loss_fn: str, loss_fn_config: dict | None = None) -> Future:
        return await _submit_async(
            "forward", self._handle, _fwd_bwd_output,
            data=[_datum_json(d) for d in data], loss_fn=loss_fn, loss_fn_config=loss_fn_config,
        )

    def forward_backward_custom(self, *args, **kwargs):
        raise NotImplementedError(
            "custom loss functions are not supported here; use one of the built-in "
            "loss_fn values: cross_entropy, importance_sampling, ppo, cispo, dro"
        )

    forward_backward_custom_async = forward_backward_custom

    def optim_step(self, adam_params: types.AdamParams) -> Future:
        return _submit(
            "optim_step", self._handle, types.OptimStepResponse.model_validate,
            adam_params=adam_params.model_dump(mode="json"),
        )

    async def optim_step_async(self, adam_params: types.AdamParams) -> Future:
        return await _submit_async(
            "optim_step", self._handle, types.OptimStepResponse.model_validate,
            adam_params=adam_params.model_dump(mode="json"),
        )

    # checkpoints

    def save_state(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False, **_: Any) -> Future:
        return _submit(
            "save_state", self._handle, types.SaveWeightsResponse.model_validate,
            name=name, ttl_seconds=ttl_seconds, overwrite=overwrite,
        )

    async def save_state_async(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False, **_: Any) -> Future:
        return await _submit_async(
            "save_state", self._handle, types.SaveWeightsResponse.model_validate,
            name=name, ttl_seconds=ttl_seconds, overwrite=overwrite,
        )

    def load_state(self, path: str, **_: Any) -> Future:
        return _submit("load_state", self._handle, types.LoadWeightsResponse.model_validate, path=path)

    async def load_state_async(self, path: str, **_: Any) -> Future:
        return await _submit_async("load_state", self._handle, types.LoadWeightsResponse.model_validate, path=path)

    def load_state_with_optimizer(self, path: str, **_: Any) -> Future:
        return _submit(
            "load_state_with_optimizer", self._handle, types.LoadWeightsResponse.model_validate, path=path
        )

    async def load_state_with_optimizer_async(self, path: str, **_: Any) -> Future:
        return await _submit_async(
            "load_state_with_optimizer", self._handle, types.LoadWeightsResponse.model_validate, path=path
        )

    def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None, **_: Any) -> Future:
        return _submit(
            "save_weights_for_sampler", self._handle, types.SaveWeightsForSamplerResponse.model_validate,
            name=name, ttl_seconds=ttl_seconds,
        )

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None, **_: Any) -> Future:
        return await _submit_async(
            "save_weights_for_sampler", self._handle, types.SaveWeightsForSamplerResponse.model_validate,
            name=name, ttl_seconds=ttl_seconds,
        )

    def save_weights_and_get_sampling_client(self, name: str | None = None, **_: Any) -> "SamplingClient":
        created = _call("save_weights_and_get_sampling_client", handle=self._handle, name=name)
        return SamplingClient(created["handle"])

    async def save_weights_and_get_sampling_client_async(self, name: str | None = None, **_: Any) -> "SamplingClient":
        return self.save_weights_and_get_sampling_client(name)

    def create_sampling_client(self, model_path: str, **_: Any) -> "SamplingClient":
        created = _call("training_create_sampling_client", handle=self._handle, model_path=model_path)
        return SamplingClient(created["handle"])

    async def create_sampling_client_async(self, model_path: str, **_: Any) -> "SamplingClient":
        return self.create_sampling_client(model_path)

    # cost planning

    def estimate_forward_backward(self, data) -> float:
        """Exact cost in USD of `forward_backward(data, ...)`."""
        return _call(
            "estimate", op="forward_backward", handle=self._handle, data=[_datum_json(d) for d in data]
        )["estimated_usd"]

    def estimate_forward(self, data) -> float:
        """Exact cost in USD of `forward(data, ...)`."""
        return _call(
            "estimate", op="forward", handle=self._handle, data=[_datum_json(d) for d in data]
        )["estimated_usd"]


class SamplingClient:
    def __init__(self, handle: str):
        self._handle = handle

    def get_base_model(self) -> str:
        return _call("get_base_model", handle=self._handle)

    async def get_base_model_async(self) -> str:
        return self.get_base_model()

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer(self._handle)

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Future:
        return _submit(
            "sample", self._handle, _sample_response,
            prompt=prompt.model_dump(mode="json"),
            num_samples=num_samples,
            sampling_params=sampling_params.model_dump(mode="json"),
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    async def sample_async(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        future = await _submit_async(
            "sample", self._handle, _sample_response,
            prompt=prompt.model_dump(mode="json"),
            num_samples=num_samples,
            sampling_params=sampling_params.model_dump(mode="json"),
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )
        return await future

    def compute_logprobs(self, prompt: types.ModelInput) -> Future:
        return _submit("compute_logprobs", self._handle, lambda x: x, prompt=prompt.model_dump(mode="json"))

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> list[float | None]:
        future = await _submit_async(
            "compute_logprobs", self._handle, lambda x: x, prompt=prompt.model_dump(mode="json")
        )
        return await future

    def estimate_sample(
        self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams
    ) -> float:
        """Worst-case cost in USD of `sample(...)`, assuming every sample runs to max_tokens."""
        return _call(
            "estimate", op="sample", handle=self._handle,
            prompt=prompt.model_dump(mode="json"),
            num_samples=num_samples,
            sampling_params=sampling_params.model_dump(mode="json"),
        )["estimated_usd"]


# -- JSON conversions ---------------------------------------------------------


def _tensor_json(td: types.TensorData) -> dict:
    return {
        "data": td.data,
        "dtype": td.dtype,
        "shape": td.shape,
        "sparse_crow_indices": td.sparse_crow_indices,
        "sparse_col_indices": td.sparse_col_indices,
    }


def _datum_json(d: types.Datum) -> dict:
    return {
        "model_input": d.model_input.model_dump(mode="json"),
        "loss_fn_inputs": {k: _tensor_json(v) for k, v in d.loss_fn_inputs.items()},
    }


def _sample_response(r: dict) -> types.SampleResponse:
    prompt_logprobs = r["prompt_logprobs"]
    return types.SampleResponse(
        sequences=[
            types.SampledSequence(
                stop_reason=s["stop_reason"],
                sequence_id=s["sequence_id"],
                tokens_np=np.asarray(s["tokens"], dtype=np.int64),
                logprobs_np=None if s["logprobs"] is None else np.asarray(s["logprobs"], dtype=np.float32),
            )
            for s in r["sequences"]
        ],
        prompt_logprobs_np=None
        if prompt_logprobs is None
        else np.asarray([np.nan if lp is None else lp for lp in prompt_logprobs], dtype=np.float32),
        prompt_cache_hit_tokens=r["prompt_cache_hit_tokens"],
        _topk_prompt_logprobs_list=r["topk_prompt_logprobs"],
    )


def _fwd_bwd_output(r: dict) -> types.ForwardBackwardOutput:
    return types.ForwardBackwardOutput(
        loss_fn_output_type=r["loss_fn_output_type"],
        loss_fn_outputs=[{k: types.TensorData(**v) for k, v in out.items()} for out in r["loss_fn_outputs"]],
        metrics=r["metrics"],
    )
