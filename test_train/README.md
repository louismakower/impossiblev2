generate the data (committed, so only needed after changing `generate.py`)
`uv run test_train/generate.py`

build the agent image
`docker build -t test_train:latest test_train/`

run the task
`inspect eval test_train/test_train.py --model openrouter/moonshotai/kimi-k3`
