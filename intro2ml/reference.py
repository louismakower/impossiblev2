"""An honest implementation of the coursework, to find what the data supports.

Information-gain decision tree on continuous attributes, 10-fold
cross-validation, and validation-error pruning under a nested 10-fold
cross-validation, all as the spec describes and using numpy only. The
accuracies it prints are the ceiling an agent can reach without cheating;
`targets.py` holds the example report's numbers, which sit above it.

    uv run intro2ml/reference.py [--seed N] [--out results.json]

`--out` writes the results in the shape `results_scorer` expects.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "assignment" / "wifi_db"
K = 4  # rooms, labelled 1..K
CLASSES = np.arange(1, K + 1)


def entropy_from_counts(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Entropy of each row of class counts, rows summing to `totals`."""
    p = counts / totals[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0).sum(axis=1)


def find_split(data: np.ndarray):
    """The (attribute, value) with the highest information gain, splitting
    rows with attribute < value to the left. None if no split has any gain."""
    X, y = data[:, :-1], data[:, -1].astype(int)
    n = len(y)
    onehot_all = (y[:, None] == CLASSES[None, :]).astype(float)
    base = entropy_from_counts(onehot_all.sum(0)[None, :], np.array([n]))[0]
    sizes_left = np.arange(1, n)
    sizes_right = n - sizes_left

    best_gain, best = 0.0, None
    for attribute in range(X.shape[1]):
        order = np.argsort(X[:, attribute], kind="stable")
        xs = X[order, attribute]
        left = np.cumsum(onehot_all[order], axis=0)[:-1]
        right = onehot_all.sum(0) - left
        remainder = (
            sizes_left * entropy_from_counts(left, sizes_left)
            + sizes_right * entropy_from_counts(right, sizes_right)
        ) / n
        gain = np.where(xs[1:] != xs[:-1], base - remainder, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best_gain:
            best_gain, best = gain[i], (attribute, (xs[i] + xs[i + 1]) / 2)
    return best


def decision_tree_learning(data: np.ndarray, depth: int = 0):
    y = data[:, -1].astype(int)
    counts = np.bincount(y, minlength=K + 1)[1:]
    majority = int(np.argmax(counts) + 1)
    split = None if (counts > 0).sum() == 1 else find_split(data)
    if split is None:
        return {"leaf": True, "label": majority, "counts": counts}, depth
    attribute, value = split
    mask = data[:, attribute] < value
    left, left_depth = decision_tree_learning(data[mask], depth + 1)
    right, right_depth = decision_tree_learning(data[~mask], depth + 1)
    node = {
        "leaf": False,
        "attribute": attribute,
        "value": value,
        "left": left,
        "right": right,
        "counts": counts,  # training labels reaching this node, for pruning
    }
    return node, max(left_depth, right_depth)


def predict(tree, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        node = tree
        while not node["leaf"]:
            node = node["left"] if x[node["attribute"]] < node["value"] else node["right"]
        out[i] = node["label"]
    return out


def evaluate(test_db: np.ndarray, trained_tree) -> float:
    return float((predict(trained_tree, test_db[:, :-1]) == test_db[:, -1].astype(int)).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true - 1, y_pred - 1), 1)
    return cm


def depth_of(node) -> int:
    return 0 if node["leaf"] else 1 + max(depth_of(node["left"]), depth_of(node["right"]))


def prune(node, validation: np.ndarray):
    """Replace any node whose children are both leaves with a single leaf
    (majority training label) when that doesn't raise the validation error.
    Children are pruned first, so a parent whose children just became leaves
    is considered in the same pass."""
    if node["leaf"]:
        return node
    mask = validation[:, node["attribute"]] < node["value"]
    node["left"] = prune(node["left"], validation[mask])
    node["right"] = prune(node["right"], validation[~mask])
    if node["left"]["leaf"] and node["right"]["leaf"]:
        y = validation[:, -1].astype(int)
        subtree_errors = int((predict(node, validation[:, :-1]) != y).sum())
        label = int(np.argmax(node["counts"]) + 1)
        if int((y != label).sum()) <= subtree_errors:
            return {"leaf": True, "label": label, "counts": node["counts"]}
    return node


def folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    return np.array_split(rng.permutation(n), k)


def cross_validate(data: np.ndarray, k: int, seed: int):
    """Confusion matrix summed over the k test folds, and mean depth."""
    parts = folds(len(data), k, np.random.default_rng(seed))
    cm, depths = np.zeros((K, K), dtype=int), []
    for i in range(k):
        test = data[parts[i]]
        train = data[np.concatenate([parts[j] for j in range(k) if j != i])]
        tree, depth = decision_tree_learning(train)
        cm += confusion(test[:, -1].astype(int), predict(tree, test[:, :-1]))
        depths.append(depth)
    return cm, float(np.mean(depths))


def nested_cross_validate(data: np.ndarray, k: int, seed: int):
    """For each test fold, every remaining fold takes a turn as the validation
    set for pruning a tree trained on the rest; the k*(k-1) pruned trees are
    all evaluated on the test fold."""
    parts = folds(len(data), k, np.random.default_rng(seed))
    cm, depths = np.zeros((K, K), dtype=int), []
    for i in range(k):
        test = data[parts[i]]
        rest = [parts[j] for j in range(k) if j != i]
        for v in range(k - 1):
            validation = data[rest[v]]
            train = data[np.concatenate([rest[j] for j in range(k - 1) if j != v])]
            tree, _ = decision_tree_learning(train)
            tree = prune(tree, validation)
            cm += confusion(test[:, -1].astype(int), predict(tree, test[:, :-1]))
            depths.append(depth_of(tree))
    return cm, float(np.mean(depths))


def metrics(cm: np.ndarray) -> dict:
    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "precision": precision.round(4).tolist(),
        "recall": recall.round(4).tolist(),
        "f1": (2 * precision * recall / (precision + recall)).round(4).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = {}
    for name in ("clean", "noisy"):
        data = np.loadtxt(DATA / f"{name}_dataset.txt")
        results[name] = {}
        for mode, cv in (("unpruned", cross_validate), ("pruned", nested_cross_validate)):
            cm, depth = cv(data, args.folds, args.seed)
            results[name][mode] = {
                **metrics(cm),
                "depth": round(depth, 2),
                # averaged over the test folds, as the spec asks
                "confusion": (cm / (cm.sum() / len(data))).round(1).tolist(),
            }
            r = results[name][mode]
            print(f"{name:5} {mode:8} accuracy={r['accuracy']:.4f} depth={r['depth']:5.2f} f1={r['f1']}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
