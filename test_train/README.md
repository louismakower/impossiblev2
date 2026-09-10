generate the data (committed, so only needed after changing `generate.py`)
`uv run test_train/generate.py`

build the agent image
`docker build -t test_train:latest test_train/`

run the task
`inspect eval test_train/test_train.py --model openrouter/moonshotai/kimi-k3`

## variants

`variant.py` holds the realism axis of the grid. The realistic variant needs its own image, with a user and home directory seeded from `home/`:

`docker build -t test_train-realistic:latest -f test_train/Dockerfile.realistic test_train/`

then pick the variant with a task arg:

`inspect eval test_train/test_train.py -T variant=realistic --model ...`

`run.py` runs both into the same log directory.
