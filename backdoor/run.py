"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run backdoor/run.py > logs/backdoor.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from backdoor_hosted import backdoor_hosted
from common.scanner import awareness, honesty
from scanner import cheating_category, target_emitted

eval(
    backdoor_hosted(),
    model=[
        # "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
    ],
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}},
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    scanner=ScannerConfig(
        scanners=[
            # cheating_category(), target_emitted(),
            honesty(), awareness()
        ],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/backdoor",
)