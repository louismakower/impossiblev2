"""Run the task. To leave it running after closing the shell:

    nohup uv run tinker_finetune/run.py > logs/tinker_finetune.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.agent import MAX_REASONING_TOKENS
from common.scanner import awareness
from tinker_finetune import tinker_finetune

eval(
    tinker_finetune(),
    model=[
        "openrouter/moonshotai/kimi-k3",
        # "openrouter/z-ai/glm-5.3",
        # "openrouter/openai/gpt-6-astra",
        # "openrouter/anthropic/claude-fable-5.1"
    ],
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}},
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    scanner=ScannerConfig(
        scanners=[awareness()],
        model="openrouter/moonshotai/kimi-k3",
    ),
    log_dir="logs/tinker_finetune",
    reasoning_tokens=MAX_REASONING_TOKENS,
    epochs=1,
    # each sample spends real money on Tinker, up to `CAP_USD`
    max_sandboxes=2,
)
