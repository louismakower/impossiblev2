start the server
`docker compose -f backdoor/server/compose.yaml up -d --wait`
confirm with: `docker ps` - make sure it says healthy

then build the agent image:
`docker build -t backdoor:latest backdoor/`

test a container has access to it:
`docker run --rm --network qwen curlimages/curl:latest -s http://vllm:8000/v1/models`

run the task:
`inspect eval backdoor/backdoor.py --model openrouter/deepseek/deepseek-v4-flash`