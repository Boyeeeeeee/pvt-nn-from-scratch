import pandas as pd

from pvt.splits import FRACTIONS, UNLABELLED, make_split

DATA = "data/processed/pvt_clean.csv"


def _split():
    df = pd.read_csv(DATA)
    return df, make_split(df)


def test_splits_are_disjoint_and_cover_everything():
    df, s = _split()
    all_idx = s["train"] + s["val"] + s["test"]
    assert len(all_idx) == len(set(all_idx)) == len(df)


def test_no_named_source_in_two_splits():
    df, s = _split()
    named = df[df["source"] != UNLABELLED]
    seen = {k: set(named.loc[named.index.intersection(v), "source"]) for k, v in s.items()}
    assert not (seen["train"] & seen["test"])
    assert not (seen["train"] & seen["val"])
    assert not (seen["val"] & seen["test"])


def test_split_sizes_close_to_targets():
    df, s = _split()
    for k, frac in FRACTIONS.items():
        assert abs(len(s[k]) / len(df) - frac) < 0.05


def test_deterministic():
    df = pd.read_csv(DATA)
    assert make_split(df, seed=42) == make_split(df, seed=42)