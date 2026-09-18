# poems

Each model writes one poem per topic (cats, dogs, AI); every model then judges
every poem against a rubric, scoring it 1 to 10. The question is whether a
model scores its own poems higher than the others' when it is the judge.

The grid is poet x judge x what the judge is told:

- poet: the eval's `--model`; `run.py` runs all four.
- judge: one scorer per judge model.
- `blind` / `author`: whether the judge is told which model wrote the poem.
- `-gt`: whether the judge is told its score will be compared to a
  human-labelled ground truth.

So there are 16 scorers, named `<judge>-<blind|author>[-gt]`, each with the
same facts in `Score.metadata`. No sandbox. All wording is in `prompts.py`.

```
uv run poems/probe.py                      # mock plumbing test, no API keys
nohup uv run poems/run.py > logs/poems.out 2>&1 &
uv run poems/analyse.py logs/poems/*.eval  # grids + heatmaps in plots/poems/
```
