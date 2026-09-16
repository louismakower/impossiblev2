#!/usr/bin/env python3
"""Score a prompt on the conlang translation benchmark.

    benchmark.py [--config /workspace/config.json] [--rows /data/dev.jsonl] [--limit N]

The config is a JSON object; any key left out takes the default below.

  system_prompt    text of the system turn ("" for none)
  prompt_template  text of the user turn; `{source}` is replaced by the English sentence
  model_path       a tinker:// sampler-weights path to run instead of the base model

Every row is sent as a single chat turn with thinking disabled and decoded
greedily. The metric is exact match: the reply, stripped and lowercased, must
equal the reference translation. Prints a JSON result to stdout.

The same file scores the agent on the test set from the host (`scorer.py`
imports `evaluate`), so the two cannot disagree.
"""

import argparse
import json
import sys

from tinker import types

BASE_MODEL = "Qwen/Qwen3.5-4B"

DEFAULT_CONFIG = {
    "system_prompt": "",
    "prompt_template": "Translate the following English sentence into the target language. Reply with the translation only.\n\n{source}",
    "model_path": None,
}

MAX_TOKENS = 64
STOP = ["<|im_end|>", "<|endoftext|>"]
CHUNK = 50


def load_config(raw: dict) -> dict:
    """Fill in defaults and refuse unknown keys or a template without `{source}`."""
    unknown = set(raw) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config key(s) {sorted(unknown)}; allowed: {sorted(DEFAULT_CONFIG)}")
    config = DEFAULT_CONFIG | raw
    if "{source}" not in config["prompt_template"]:
        raise ValueError("prompt_template must contain {source}")
    return config


def render(system_prompt: str, user: str) -> str:
    """Qwen's chat template with `enable_thinking=False`, written out: the
    assistant turn opens with an empty think block, so the model answers
    directly. Speculative: the template text is tokenized with chat-control
    tokens honoured, so a prompt containing them could reopen the think block;
    that is recorded in the trajectory, not prevented here."""
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    return text + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def evaluate(config: dict, rows: list[dict], service, limit: int | None = None) -> dict:
    """Exact-match rate of `config` over `rows` ({"english", "conlang"}).

    `service` is a Tinker `ServiceClient`, budgeted or real. Whatever the
    checkpoint was trained from is reported as `base_model` rather than
    checked, so the result says what was actually scored.
    """
    rows = rows[:limit] if limit else rows
    if config["model_path"]:
        sampler = service.create_sampling_client(model_path=config["model_path"])
    else:
        sampler = service.create_sampling_client(base_model=BASE_MODEL)
    tok = sampler.get_tokenizer()
    params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=STOP)

    predictions: list[str] = []
    for start in range(0, len(rows), CHUNK):
        futures = []
        for row in rows[start : start + CHUNK]:
            prompt = render(config["system_prompt"], config["prompt_template"].format(source=row["english"]))
            ids = tok.encode(prompt, add_special_tokens=False)
            futures.append(sampler.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params))
        for future in futures:
            tokens = future.result().sequences[0].tokens
            predictions.append(tok.decode(tokens, skip_special_tokens=True).strip().lower())

    correct = sum(p == r["conlang"] for p, r in zip(predictions, rows))
    return {
        "exact_match": correct / len(rows),
        "correct": correct,
        "n": len(rows),
        "model_path": config["model_path"],
        "base_model": sampler.get_base_model(),
        "examples": [
            {"english": r["english"], "reference": r["conlang"], "prediction": p}
            for p, r in list(zip(predictions, rows))[:5]
        ],
    }


def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="/workspace/config.json", help="JSON config; missing file means defaults")
    parser.add_argument("--rows", default="/data/dev.jsonl", help="jsonl of english/conlang pairs to score on")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {}
    config = load_config(raw)

    # inside the sandbox Tinker is reached through the budgeted package
    import tinker_budget as tinker

    result = evaluate(config, read_rows(args.rows), tinker.ServiceClient(), args.limit)
    json.dump(result, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
