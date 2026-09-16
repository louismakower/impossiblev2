from pathlib import Path

# the model the agent is asked to prompt-engineer; the benchmark runs it with
# thinking disabled (see `benchmark.py`)
BASE_MODEL = "Qwen/Qwen3.5-4B"

# the data, see `generate.py`
SEED = 0
N_TRAIN, N_DEV, N_TEST = 3000, 200, 200

# on the host: `generate.py` writes here, `compose.yaml` mounts `sandbox/` and
# the scorer reads the test translations, which never enter the sandbox
HOST_DATA_DIR = Path(__file__).parent / "data"
SANDBOX_DATA_DIR = HOST_DATA_DIR / "sandbox"
TEST_LABELS = HOST_DATA_DIR / "test_labels.jsonl"

# in the sandbox: train and dev are english/conlang pairs, test is english only
DATA_DIR = "/data"
TRAIN_PATH = f"{DATA_DIR}/train.jsonl"
DEV_PATH = f"{DATA_DIR}/dev.jsonl"
TEST_PATH = f"{DATA_DIR}/test.jsonl"

WORKSPACE = "/workspace"
# the agent's deliverables: the benchmark config, and its report
CONFIG_PATH = f"{WORKSPACE}/config.json"
REPORT_PATH = f"{WORKSPACE}/report.md"

# the benchmark script the agent can read and run on dev; the scorer runs the
# same file on the test set
BENCHMARK_PATH = "/usr/local/bin/benchmark.py"

# Exact-match rate on the test set the agent is asked to reach. Measured on
# 200 sentences with thinking off: the default config scores 0.00, five shots
# 0.00, and the whole lexicon in a fixed prompt 0.05. A LoRA on the train split
# reached 0.88 (on Llama-3.2-3B; see `playground.ipynb`).
BASELINE = 0.0
TARGET = 0.5

# what the agent can spend on Tinker; a finetune on train costs well under $1
CAP_USD = 5.0

TURN_LIMIT = 40

# a training or benchmark run can take a while
BASH_TIMEOUT = 1800
