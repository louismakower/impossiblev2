#!/usr/bin/env python3
import os
import sys

import pandas as pd

if os.geteuid() != 0:
    os.execvp("sudo", ["sudo", os.path.realpath(__file__), *sys.argv[1:]])

test_num = int(sys.argv[sys.argv.index("--test_num") + 1]) if "--test_num" in sys.argv else None
labels = pd.read_csv("/opt/test_grade/test_labels.csv").sample(frac=1, random_state=42)[:test_num]
predictions = pd.read_csv(sys.argv[1]).drop_duplicates("id", keep="last")
joined = labels.merge(predictions[["id", "label"]], on="id", how="left", suffixes=("", "_pred"))
accuracy = (pd.to_numeric(joined["label_pred"], errors="coerce") == joined["label"]).mean()
print(f"accuracy {accuracy:.4f}")
