"""Group train/val/test split by `source`, fixed seed, saved to data/processed/splits.json.

Why groups: records from the same paper/field share measurement style and often come in
near-identical families. A random row split puts siblings in both train and test and
inflates the scores. Here every named source lands entirely in one split.

Exception: rows whose S/N is just a number carry no usable source label (they come from a
mix of papers). Treating them as one group would drop 250+ rows into a single split, so
each of those rows is its own group instead. That part of the split is therefore random
rather than source-grouped; the README should say so.

Usage (from the repo root):  python src/pvt/splits.py
"""
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
UNLABELLED = "numeric_id"


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Group label per row: the source, except unlabelled rows get one group each."""
    groups = df["source"].astype(str).copy()
    unl = groups == UNLABELLED
    groups[unl] = [f"{UNLABELLED}_{i}" for i in df.index[unl]]
    return groups


def make_split(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Greedy group assignment: shuffle groups with a fixed seed, place big groups first,
    and give each group to whichever split is furthest below its target size.
    Deterministic for a given seed."""
    groups = assign_groups(df)
    sizes = groups.value_counts().to_dict()
    names = list(sizes)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -sizes[g])  # stable sort keeps the seeded order within ties

    n = len(df)
    target = {k: v * n for k, v in FRACTIONS.items()}
    filled = {k: 0 for k in FRACTIONS}
    owner = {}
    for g in names:
        deficit = {k: target[k] - filled[k] for k in FRACTIONS}
        k = max(deficit, key=deficit.get)
        owner[g] = k
        filled[k] += sizes[g]

    split_of_row = groups.map(owner)
    return {k: sorted(int(i) for i in df.index[split_of_row == k]) for k in FRACTIONS}


def save_split(split: dict, path: str, seed: int = SEED) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "fractions": FRACTIONS, **split}, f)


def load_split(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {k: d[k] for k in FRACTIONS}


if __name__ == "__main__":
    data = pd.read_csv("data/processed/pvt_clean.csv")
    split = make_split(data)
    save_split(split, "data/processed/splits.json")
    for name, idx in split.items():
        sub = data.loc[idx]
        named = sub[sub["source"] != UNLABELLED]
        print(f"{name:5s} rows={len(idx):4d}  named-source rows={len(named):4d}  sources={named['source'].nunique()}")
    print("saved -> data/processed/splits.json")
