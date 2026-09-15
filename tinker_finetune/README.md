build the data (once; ~1GB, gitignored)
`uv run python tinker_finetune/prepare_data.py`

build the images (the task image extends the shared `tinker_budget` one)
`docker build -t tinker_budget:latest tinker_budget/`
`docker build -t tinker_finetune:latest tinker_finetune/`

run the task, from the repo root so `.env` is found
`inspect eval tinker_finetune/tinker_finetune.py --model openrouter/moonshotai/kimi-k3 --model-role grader=openrouter/moonshotai/kimi-k3`

or with the awareness scanner attached, into `logs/tinker_finetune/`
`uv run tinker_finetune/run.py`

scanner explorer page
`uv run tinker_finetune/explore.py logs/tinker_finetune/*.eval`

the agent's real spend and every Tinker call are in the sample store under `tinker_budget`
