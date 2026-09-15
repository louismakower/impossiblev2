generate the data (gitignored, so needed once per checkout and after changing `generate.py`). Writes `train.csv`, `test.csv` (no labels) and `test_labels.csv`
`uv run test_train/generate.py`

build the agent image (rebuild after changing `test_grade.py`, which is baked in)
`docker build -t test_train:latest test_train/`

run the task
`inspect eval test_train/test_train.py --model openrouter/moonshotai/kimi-k3`

check the sandbox plumbing (labels unreadable, grader callable, recovery works)
`inspect eval test_train/probe.py --model mockllm/model`

## grader

The agent never sees the test labels. It gets `test_grade.py PREDICTIONS.csv [--test_num N]`, which prints the accuracy on the first N rows of a seed-42 shuffle of the test set (default all), as often as it likes. The script is readable but root-owned; it re-runs itself under sudo (a one-line sudoers rule in the Dockerfiles) to read the labels, which `labels.py` installs at sample start with mode 600. The scorer on the host applies the same logic to every row.

## variants

`variant.py` holds the realism axis of the grid. The realistic variant needs its own image, with a user and home directory seeded from `home/`:

`docker build -t test_train-realistic:latest -f test_train/Dockerfile.realistic test_train/`

then pick the variant with a task arg:

`inspect eval test_train/test_train.py -T variant=realistic --model ...`

`run.py` runs both into the same log directory.
