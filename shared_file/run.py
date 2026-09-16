"""Run the task. To leave it running after closing the shell:

    nohup uv run shared_file/run.py > logs/shared_file.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.agent import MAX_REASONING_TOKENS
from common.scanner import awareness
from scanner import behaviour
from shared_file import shared_file

eval(
    shared_file(),
    model=[
        "openrouter/moonshotai/kimi-k3",
        # "openrouter/z-ai/glm-5.3",
    ],
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}},
    model_roles={"grader": "openrouter/moonshotai/kimi-k3"},
    scanner=ScannerConfig(
        scanners=[awareness(), behaviour()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/shared_file",
    reasoning_tokens=MAX_REASONING_TOKENS,
    # a sample is two agents making up to ~100 model calls each
    epochs=4,
    max_sandboxes=4,
)
