"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run test_train/run_eval_set.py > logs/test_train-grid.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval_set

from common.scanner import awareness, honesty
from test_train import test_train

test_train_csv = test_train(submission="csv")
test_train_report = test_train(submission="report")
test_train_report_trajectory = test_train(submission="report_trajectory")

eval_set(
    [test_train_csv, test_train_report, test_train_report_trajectory],
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
    log_dir="logs/test_train-grid",
)
