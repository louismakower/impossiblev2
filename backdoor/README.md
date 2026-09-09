download the local model into the HF cache
`hf download Qwen/Qwen3.5-4B`

start the server
`docker compose -f backdoor/server/compose.yaml up -d --wait`
confirm with: `docker ps` - make sure it says healthy

then build the agent image:
`docker build -t backdoor:latest backdoor/`

test a container has access to it:
`docker run --rm --network qwen curlimages/curl:latest -s http://llm:8000/v1/models`

run the task:
`inspect eval backdoor/backdoor.py --model openrouter/deepseek/deepseek-v4-flash`

## variants

`variant.py` holds the realism axis of the grid. The baseline runs in a bare `python:3.13-slim` container; the realistic variant needs its own image, with a user and home directory seeded from `home/`:

`docker build -t backdoor-realistic:latest -f backdoor/Dockerfile.realistic backdoor/`

then pick the variant with a task arg:

`inspect eval backdoor/backdoor_hosted.py -T variant=realistic --model ...`

`run.py` runs both into the same log directory.