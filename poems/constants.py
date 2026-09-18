"""Names shared by the poems task."""

TOPICS = ["cats", "dogs", "AI"]

# short name -> model id. The same four models act as poets (the task's
# `--model`) and as judges (one scorer each), so the grid is poet x judge.
MODELS = {
    "kimi": "openrouter/moonshotai/kimi-k3",
    "glm": "openrouter/z-ai/glm-5.3",
    "astra": "openrouter/openai/gpt-6-astra",
    "fable": "anthropic/claude-fable-5-1",
}

SCORE_MIN = 1
SCORE_MAX = 10
