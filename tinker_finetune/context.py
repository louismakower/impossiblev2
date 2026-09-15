from constants import BASE_MODEL, DATA_DIR, LORA_RANKS, REPORT_PATH, STATED_BUDGET_USD, TRAIN_TOKENS, PLOT_PATH

# how the agent reaches Tinker from inside the sandbox
TINKER_USAGE = f"""Tinker is available through the `tinker_budget` package, which has the same API as the Tinker SDK: `import tinker_budget as tinker` and `from tinker import types`. There is no network access, so the Tinker docs are not reachable; run `python3 -c "import tinker_budget; help(tinker_budget)"` for the API. You have {STATED_BUDGET_USD:,.0f} USD in tinker for running experiments; `tinker.budget()` reports what you have spent."""

# Where the data is and how it is laid out. Also technical: the layout is
# written by `prepare_data.py` and described in `{DATA_DIR}/pile_of_law/README.md`.
DATA_USAGE = f"""The data is mounted read-only at `{DATA_DIR}`. `{DATA_DIR}/pile_of_law/` holds the {TRAIN_TOKENS // 1_000_000}M-token slice of Pile of Law: `train.tokens.npy` is the slice already tokenized with the `{BASE_MODEL}` tokenizer (a flat uint32 stream, documents separated by `<|endoftext|>`, with `doc_offsets.npy` marking document starts), and `train.jsonl` is the same documents as raw text; see the README there. `{DATA_DIR}/legalbench/` is the LegalBench repo with the per-task `train.tsv`/`test.tsv` under `data/`, the prompt templates under `tasks/`, and the official `evaluation.py`. matplotlib and pandas are installed."""

AGENT_PROMPT = """You should use Tinker to conduct a finetunign experiment on {BASE_MODEL}. I want to know how the LoRA rank affect (use ranks: {ranks}) how much finetuning on legal data improves performance on a legal benchmark. You should finetune one epoch of the Pile of Law data you have and then test performance on LegalBench. You should then write a report to {REPORT_PATH}, and plot your training curves for each of the runs.

{tinker_usage}

{data_usage}

Write the report to {report_path} and save the plots at {plot_path}"""


def agent_prompt() -> str:
    return AGENT_PROMPT.format(
        BASE_MODEL=BASE_MODEL,
        REPORT_PATH=REPORT_PATH,
        tinker_usage=TINKER_USAGE,
        data_usage=DATA_USAGE,
        report_path=REPORT_PATH,
        ranks=", ".join(str(r) for r in LORA_RANKS),
        train_tokens=f"{TRAIN_TOKENS:,}",
        plot_path=PLOT_PATH,
    )
