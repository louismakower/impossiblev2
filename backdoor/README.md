The target LLM is reached over OpenRouter through the `llm` tool, so the sandbox has no network and needs no image build for the baseline.

run the task:
`inspect eval backdoor/backdoor_hosted.py --model openrouter/moonshotai/kimi-k3`

check the tool and approver plumbing:
`inspect eval backdoor/llm_tool_probe.py --model mockllm/model`

## variants

`variant.py` holds the realism axis of the grid. The baseline runs in a bare `python:3.13-slim` container; the realistic variant needs its own image, with a user and home directory seeded from `home/`:

`docker build -t backdoor-realistic:latest -f backdoor/Dockerfile.realistic backdoor/`

then pick the variant with a task arg:

`inspect eval backdoor/backdoor_hosted.py -T variant=realistic --model ...`

`run.py` runs both into the same log directory.