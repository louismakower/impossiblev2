"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run memleak/run.py > logs/memleak.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.scanner import awareness
from common.agent import MAX_REASONING_TOKENS
from memleak import memleak

# grading cells to run; None runs the whole grid
GRADINGS = None

eval(
    [memleak(variant="baseline", gradings=GRADINGS)],
    model=[
        "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
    ],
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}},
    model_roles={"grader": "openrouter/moonshotai/kimi-k3"},
    scanner=ScannerConfig(
        scanners=[awareness()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/memleak",
    reasoning_tokens=MAX_REASONING_TOKENS,
    epochs=1,
    max_sandboxes=8,
    fail_on_error=False,
)
