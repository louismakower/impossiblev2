"""Plumbing test with mock models: a mock poet and a mock judge that always
answers `SCORE: 7`, plus one that never gives a score, to check the parser and
the unscored path. No API keys needed.

    uv run poems/probe.py
"""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from poems import poems


def always(text: str):
    """A mock judge that gives the same reply however often it is asked.

    A callable rather than an infinite iterator: `custom_outputs` gets
    serialised with the model args, and listing `itertools.repeat` ate the
    machine's memory.
    """
    output = ModelOutput.from_content("mockllm/model", text)
    return get_model("mockllm/model", custom_outputs=lambda *_: output)


judges = {"seven": always("Fine poem.\n\nSCORE: 7"), "mute": always("No score here.")}

[log] = eval(
    poems(judges=judges),
    model="mockllm/model",
    log_dir="logs/poems_probe",
    display="plain",
)

assert log.status == "success", log.error
for sample in log.samples or []:
    for name, score in sample.scores.items():
        if name.startswith("seven"):
            assert score.value == 7, (name, score)
        else:
            assert score.value != score.value, (name, score)  # NaN: unscored
        assert score.metadata["poet"] == "mockllm/model"
print("ok:", sorted(log.samples[0].scores))
