"""Runs inside the sandbox. Exercises every kind of call once."""

import asyncio

import tinker_budget as tinker
from tinker import types

MODEL = "Qwen/Qwen3-8B"

print("budget at start", tinker.budget())
svc = tinker.ServiceClient()

try:
    svc.create_lora_training_client(base_model="nope/NotAModel")
except tinker.UnknownModel as e:
    print("unknown model refused:", e)

training = svc.create_lora_training_client(base_model=MODEL)
info = training.get_info()
print("info", info.model_data.model_name, info.lora_rank)

tok = training.get_tokenizer()
ids = tok.encode("The capital of France is")
print("encoded", ids)
print("chat template", tok.apply_chat_template([{"role": "user", "content": "hi"}])[:8], "...")

data = [
    types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
    )
]
print("estimate forward_backward", training.estimate_forward_backward(data))
fb = training.forward_backward(data, "cross_entropy")
opt = training.optim_step(types.AdamParams(learning_rate=1e-5))
print("submitted both; estimated", fb.estimated_usd, opt.estimated_usd)
out = fb.result()
print("forward_backward metrics", out.metrics, "cost", fb.cost_usd)
opt.result()
print("optim_step cost", opt.cost_usd)

fwd = training.forward(data, "cross_entropy").result()
print("forward logprobs", fwd.loss_fn_outputs[0]["logprobs"].tolist()[:3], "...")

sampler = training.save_weights_and_get_sampling_client(name="smoke")
print("sampler base model", sampler.get_base_model())
prompt = types.ModelInput.from_ints(ids)
sp = types.SamplingParams(max_tokens=8, temperature=0.0)
print("estimate sample", sampler.estimate_sample(prompt, 2, sp))
fut = sampler.sample(prompt, num_samples=2, sampling_params=sp)
resp = fut.result()
for s in resp.sequences:
    print("  sampled", repr(tok.decode(s.tokens)), s.stop_reason)
print("sample cost", fut.cost_usd, "estimated", fut.estimated_usd, "cache hits", resp.prompt_cache_hit_tokens)

lp = sampler.compute_logprobs(prompt).result()
print("compute_logprobs", lp[:3], "...")


async def async_path():
    r = await sampler.sample_async(prompt, num_samples=1, sampling_params=sp)
    print("async sample", repr(tok.decode(r.sequences[0].tokens)))


asyncio.run(async_path())

try:
    sampler.sample(prompt, num_samples=1, sampling_params=types.SamplingParams(temperature=0.0))
except ValueError as e:
    print("max_tokens required:", e)

try:
    sampler.sample(prompt, num_samples=1000, sampling_params=types.SamplingParams(max_tokens=10000))
except tinker.BudgetExceeded as e:
    print("budget exceeded:", e)

try:
    training.forward_backward_custom(data, lambda d, lp: None)
except NotImplementedError as e:
    print("custom loss refused:", e)

print("budget at end", tinker.budget())
