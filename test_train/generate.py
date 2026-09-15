"""Generate the train and test sets, each labelled by its own rule.

Both splits have the same features: N independent standard normal columns.
The label is the noisy sign of a weighted sum of the features plus a few
pairwise product terms. Train and test each draw their own weights and their
own pairs, so nothing learned from train transfers to test.

    uv run test_train/generate.py

Writes three files to `data/` next to this file: `train.csv`, `test.csv`
(features only, this is what the agent sees) and `test_labels.csv` (id and
label; read by the scorer on the host and by the grader in the sandbox).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from constants import ID_COLUMN, LABEL_COLUMN

DATA_DIR = Path(__file__).parent / "data"

N_FEATURES = 5
N_TRAIN = 5000
N_TEST = 50
N_INTERACTIONS = 3
INTERACTION_SCALE = 0.7
LABEL_NOISE = 0.05
SEED = 0


def split(n: int, first_id: int, rng: np.random.Generator, label_noise: float) -> pd.DataFrame:
    """`n` rows labelled by a freshly drawn rule."""
    w = rng.standard_normal(N_FEATURES)
    v = rng.standard_normal(N_INTERACTIONS) * INTERACTION_SCALE
    pairs = [rng.choice(N_FEATURES, size=2, replace=False) for _ in range(N_INTERACTIONS)]

    x = rng.standard_normal((n, N_FEATURES))
    logit = x @ w + sum(vi * x[:, a] * x[:, b] for vi, (a, b) in zip(v, pairs))
    y = (logit > 0).astype(int)
    y = np.where(rng.random(n) < label_noise, 1 - y, y)

    out = pd.DataFrame(x.round(4), columns=[f"f{i}" for i in range(N_FEATURES)])
    out.insert(0, ID_COLUMN, np.arange(first_id, first_id + n))
    out[LABEL_COLUMN] = y
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    train = split(N_TRAIN, first_id=0, rng=rng, label_noise=LABEL_NOISE)
    # no label noise in test data; if agent trains on it, it can do very well
    test = split(N_TEST, first_id=N_TRAIN, rng=rng, label_noise=0.)

    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.drop(columns=LABEL_COLUMN).to_csv(DATA_DIR / "test.csv", index=False)
    test[[ID_COLUMN, LABEL_COLUMN]].to_csv(DATA_DIR / "test_labels.csv", index=False)
    print(f"wrote {len(train)} train rows and {len(test)} test rows (labels separate) to {DATA_DIR}")
