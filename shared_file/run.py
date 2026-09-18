"""Run the task, the whole head start × bonus factor sweep, over every model.
To leave it running after closing the shell:

    nohup uv run shared_file/run.py > logs/shared_file.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import ScannerConfig, eval

from common.agent import MAX_REASONING_TOKENS
from common.scanner import awareness
from scanner import behaviour
from shared_file import shared_file

MODELS = [
    "openrouter/moonshotai/kimi-k3",
    "openrouter/openai/gpt-6-astra",
    "openrouter/anthropic/claude-fable-5.1",
    "openrouter/z-ai/glm-5.3",
]

eval(
    shared_file(),
    model=MODELS,
    # These upstream providers serve GLM-5.3 with the closing brace dropped
    # from every non-final tool call in a multi-call turn, so Inspect rejects
    # all but the last call with a parsing error.
    model_args={"provider": {"ignore": ["Phala", "Modal", "Reka"]}},
    model_roles={"grader": "openrouter/moonshotai/kimi-k3"},
    # scanner=ScannerConfig(
    #     # scanners=[awareness(), behaviour()],
    #     model="openrouter/moonshotai/kimi-k3",
    # ),
    log_dir="logs/shared_file",
    reasoning_tokens=MAX_REASONING_TOKENS,
    # a sample is two agents making up to ~100 model calls each; the
    # containers are small (see `compose.yaml`) so many can run at once
    epochs=10,
    max_sandboxes=30,
)
