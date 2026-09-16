"""Resume errored logs from `run.py`, keeping their completed samples.

    nohup uv run backdoor/retry.py logs/backdoor/<log>.eval [...] > logs/backdoor_retry.out 2>&1 &

`eval_retry` restores the model, model_args, model_roles, task_args and epochs
from each log, and re-imports the task file so scorer changes apply to the
resumed samples. It does not restore the scanner config or log_dir, so those
are passed again here; the scanner config must match `run.py` exactly or
Inspect refuses to attach to the existing scan directory.
"""

import sys

from dotenv import load_dotenv
from inspect_ai import ScannerConfig, eval_retry

from common.scanner import awareness

# eval_retry resolves the model from the log before Inspect loads .env (eval
# loads it first), so without this it fails with "No OPENROUTER_API_KEY".
load_dotenv()

eval_retry(
    sys.argv[1:],
    log_dir="logs/backdoor",
    scanner=ScannerConfig(
        scanners=[awareness()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    fail_on_error=False,
    # run.py is still running the kimi tasks with 50 sandboxes on this machine
    max_sandboxes=25,
)
