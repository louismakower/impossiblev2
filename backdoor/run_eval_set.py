"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run backdoor/run_eval_set.py > logs/backdoor-grid.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval_set

from backdoor_hosted import backdoor_hosted
from common.scanner import awareness, honesty
from scanner import cheating_category, target_emitted

backdoor_json = backdoor_hosted(submission="json")
backdoor_tool = backdoor_hosted(submission="tool")
backdoor_report = backdoor_hosted(submission="report")
backdoor_report_trajectory = backdoor_hosted(submission="report_trajectory")

eval_set(
    [backdoor_json, backdoor_tool, backdoor_report, backdoor_report_trajectory],
    model=[
        "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
    ],
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    scanner=ScannerConfig(
        scanners=[cheating_category(), target_emitted(), honesty(), awareness()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/backdoor-grid",
)