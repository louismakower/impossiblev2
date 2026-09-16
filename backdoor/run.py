"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run backdoor/run.py > logs/backdoor.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from backdoor_hosted import backdoor_hosted
from common.scanner import awareness
from common.agent import MAX_REASONING_TOKENS
from scanner import cheating_category, target_emitted

# grading cells to run; None runs the whole grid
GRADINGS = None

eval(
    # both variants in one log directory, so a single scan can group by variant
    [
        backdoor_hosted(variant="baseline", gradings=GRADINGS),
        backdoor_hosted(variant="realistic", gradings=GRADINGS),
    ],
    model=[
        "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
        # "openrouter/openai/gpt-6-astra",
        # "openrouter/anthropic/claude-fable-5.1"
    ],
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}, "strict_tools": False},
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    scanner=ScannerConfig(
        scanners=[
            # cheating_category(), target_emitted(), honesty(),
            awareness()
        ],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/backdoor",
    reasoning_tokens=MAX_REASONING_TOKENS,
    epochs=100,
    max_sandboxes=50,
    fail_on_error=False,
)