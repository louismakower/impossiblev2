"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run test_train/run.py > logs/test_train.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.scanner import awareness, honesty
from test_train import test_train

eval(
    test_train(),
    model=[
        "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
    ],
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    scanner=ScannerConfig(
        scanners=[honesty(), awareness()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/test_train",
)
