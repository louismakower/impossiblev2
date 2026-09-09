"""Build `tinker_finetune/data/`, which compose mounts read-only at `/data`.

Run once from the repo root: `uv run python tinker_finetune/prepare_data.py`.

Pile of Law: whole documents from one shard of court opinions, streamed and
decompressed on the fly so only the part we need is downloaded, until the
stream holds at least `TRAIN_TOKENS` tokens. Documents are tokenized with the
base model's tokenizer and joined with its `<|endoftext|>` token, the
document separator Qwen uses in pretraining. The raw text goes alongside in
the same order, with the upstream dataset card.

LegalBench: the GitHub repo (prompts, scoring code, per-task READMEs) with the
dataset repo's `data/` (train/test TSVs) and task metadata dropped into it.

    data/pile_of_law/train.tokens.npy   uint32, >= TRAIN_TOKENS long
    data/pile_of_law/doc_offsets.npy    int64, start of each document in the stream
    data/pile_of_law/train.jsonl        {"url", "text", ...} per line
    data/pile_of_law/README.md, dataset_card.md
    data/legalbench/                    the GitHub repo, plus data/ and task_metadata.json
"""

import argparse
import json
import lzma
import shutil
import subprocess
import sys
from itertools import batched
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from constants import BASE_MODEL, TRAIN_TOKENS  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

PILE_OF_LAW = "https://huggingface.co/datasets/pile-of-law/pile-of-law"
PILE_OF_LAW_SHARD = "train.courtlisteneropinions.0.jsonl.xz"
LEGALBENCH_DATASET = "nguha/legalbench"
LEGALBENCH_GITHUB = "https://github.com/HazyResearch/legalbench"

SEPARATOR = "<|endoftext|>"
ENCODE_BATCH = 64


def shard_documents() -> Iterator[dict]:
    """The shard's records in order, decompressed as they download."""
    with httpx.stream("GET", f"{PILE_OF_LAW}/resolve/main/data/{PILE_OF_LAW_SHARD}", follow_redirects=True) as r:
        r.raise_for_status()
        decompressor = lzma.LZMADecompressor()
        buffer = b""
        for chunk in r.iter_bytes():
            buffer += decompressor.decompress(chunk)
            *lines, buffer = buffer.split(b"\n")
            yield from (json.loads(line) for line in lines if line)


def pile_of_law(out: Path, n_tokens: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_pretrained(BASE_MODEL)
    separator = tokenizer.token_to_id(SEPARATOR)

    chunks: list[np.ndarray] = []
    offsets: list[int] = []
    total = 0
    with open(out / "train.jsonl", "w") as jsonl:
        for batch in batched(shard_documents(), ENCODE_BATCH):
            encodings = tokenizer.encode_batch([record["text"] for record in batch], add_special_tokens=False)
            for record, encoding in zip(batch, encodings):
                offsets.append(total)
                chunks.append(np.array(encoding.ids + [separator], dtype=np.uint32))
                total += len(chunks[-1])
                jsonl.write(json.dumps(record) + "\n")
                if total >= n_tokens:
                    break
            print(f"\r{total / 1e6:,.1f}M / {n_tokens / 1e6:,.0f}M tokens, {len(offsets):,} documents", end="")
            if total >= n_tokens:
                break
    print()

    np.save(out / "train.tokens.npy", np.concatenate(chunks))
    np.save(out / "doc_offsets.npy", np.asarray(offsets, dtype=np.int64))
    (out / "dataset_card.md").write_bytes(httpx.get(f"{PILE_OF_LAW}/raw/main/README.md", follow_redirects=True).content)
    (out / "README.md").write_text(f"""The first {len(offsets):,} documents of `{PILE_OF_LAW_SHARD}` from Pile of Law
({PILE_OF_LAW}; US court opinions from CourtListener, see `dataset_card.md`),
tokenized with the `{BASE_MODEL}` tokenizer.

- `train.tokens.npy`: uint32 token ids, one flat stream of {total:,} tokens.
  Documents are separated by `{SEPARATOR}` (id {separator}).
- `doc_offsets.npy`: int64 index into the stream where each document starts.
- `train.jsonl`: the same documents in the same order as raw text, one JSON
  object per line with `url`, `text`, `created_timestamp`,
  `downloaded_timestamp`.
""")


def legalbench(out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", "--quiet", LEGALBENCH_GITHUB, str(out)], check=True)
    shutil.rmtree(out / ".git")
    snapshot_download(
        LEGALBENCH_DATASET, repo_type="dataset", local_dir=out, allow_patterns=["data/*", "task_metadata.json"]
    )
    shutil.rmtree(out / ".cache")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=TRAIN_TOKENS, help="minimum size of the Pile of Law slice")
    parser.add_argument("--skip-legalbench", action="store_true")
    args = parser.parse_args()
    pile_of_law(DATA_DIR / "pile_of_law", args.tokens)
    if not args.skip_legalbench:
        legalbench(DATA_DIR / "legalbench")
