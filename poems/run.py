"""Run the task over every poet model. To leave it running after closing the shell:

    nohup uv run poems/run.py > logs/poems.out 2>&1 &

then monitor with `inspect ctl task list`.
"""

from inspect_ai import eval

from constants import MODELS
from poems import poems

eval(
    poems(),
    model=list(MODELS.values()),
    log_dir="logs/poems",
    epochs=5,
)
