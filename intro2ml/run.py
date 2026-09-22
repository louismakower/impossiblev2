"""Run the whole grid. To leave it running after closing the shell:

    nohup uv run intro2ml/run.py > logs/intro2ml.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.scanner import awareness
from common.agent import MAX_REASONING_TOKENS
from intro2ml import intro2ml

# grading cells to run; None runs the whole grid
GRADINGS = None

eval(
    # both variants in one log directory, so a single scan can group by variant
    [
        intro2ml(variant="baseline", gradings=GRADINGS),
        intro2ml(variant="realistic", gradings=GRADINGS),
    ],
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
    log_dir="logs/intro2ml",
    reasoning_tokens=MAX_REASONING_TOKENS,
    epochs=50,
    max_sandboxes=20,
    fail_on_error=False,
)
