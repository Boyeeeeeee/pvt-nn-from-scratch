"""Ablation studies on the test split (same data, same split, same fixed hyperparameters).

Usage (repo root):  python src/pvt/ablations.py

1. Feature ablation: drop one input at a time, use raw Rs instead of log10(Rs), or add the
   Standing correlation's own estimate as an extra input.
2. Joint vs separate: one 2-output network (Pb and Bo together) vs the two single-target networks.

Fixed config for every run: hidden (64, 64), dropout 0, weight decay 1e-3, 8 seeds.
Writes results/ablation_studies.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvt.baselines import standing_bo, standing_pb  # noqa: E402
from pvt.features import Standardizer, from_target, to_target  # noqa: E402
from pvt.metrics import aare  # noqa: E402
from pvt.model import PVTNet  # noqa: E402
from pvt.splits import load_split  # noqa: E402
from pvt.train import fit, set_seed  # noqa: E402

CSV, SPLITS = "data/processed/pvt_clean.csv", "data/processed/splits.json"
CFG = {"hidden": (64, 64), "dropout": 0.0, "wd": 1e-3}
SEEDS = range(8)
VARIANTS = ["full", "no temp", "no gg", "no api", "raw rs (no log)", "+ Standing estimate"]


def features(df, variant, target):
    rs, gg, api, temp = (df[c].to_numpy(float) for c in ("rs", "gg", "api", "temp"))
    cols = {"log_rs": np.log10(rs), "rs": rs, "gg": gg, "api": api, "temp": temp}
    pick = {
        "full": ["log_rs", "gg", "api", "temp"],
        "no temp": ["log_rs", "gg", "api"],
        "no gg": ["log_rs", "api", "temp"],
        "no api": ["log_rs", "gg", "temp"],
        "raw rs (no log)": ["rs", "gg", "api", "temp"],
        "+ Standing estimate": ["log_rs", "gg", "api", "temp", "standing"],
    }[variant]
    if "standing" in pick:
        est = standing_pb(rs, gg, api, temp) if target == "pb" else standing_bo(rs, gg, api, temp)
        cols["standing"] = np.log10(est) if target == "pb" else est
    return np.column_stack([cols[c] for c in pick])


def run(df, split, x_all, targets, seed):
    """Train one network (1 or 2 outputs) and return {target: test predictions in original units}."""
    z_all = np.column_stack([to_target(df[t], t) for t in targets])
    xs = Standardizer().fit(x_all[split["train"]])
    ys = Standardizer().fit(z_all[split["train"]])

    def T(a):
        return torch.tensor(a, dtype=torch.float32)

    tr = (T(xs.transform(x_all[split["train"]])), T(ys.transform(z_all[split["train"]])))
    va = (T(xs.transform(x_all[split["val"]])), T(ys.transform(z_all[split["val"]])))
    if len(targets) == 1:
        tr, va = (tr[0], tr[1].squeeze(-1)), (va[0], va[1].squeeze(-1))
    set_seed(seed)
    model = PVTNet(x_all.shape[1], CFG["hidden"], CFG["dropout"], out_dim=len(targets))
    fit(model, tr, va, weight_decay=CFG["wd"], seed=seed)
    model.eval()
    with torch.no_grad():
        out = model(T(xs.transform(x_all[split["test"]]))).numpy().reshape(len(split["test"]), -1)
    z = ys.inverse(out)
    return {t: from_target(z[:, i], t) for i, t in enumerate(targets)}


def summarise(rows, study, variant, target, per_seed, y, named):
    for subset, mask in (("all test", np.ones(len(y), bool)), ("named-source only", named)):
        a = [aare(y[mask], p[mask]) for p in per_seed]
        rows.append({"study": study, "variant": variant, "target": target, "subset": subset,
                     "AARE_mean": np.mean(a), "AARE_std": np.std(a)})


def main():
    df = pd.read_csv(CSV)
    split = load_split(SPLITS)
    test = df.loc[split["test"]]
    named = (test["source"] != "numeric_id").to_numpy()
    rows = []

    for target in ("pb", "bo"):
        y = test[target].to_numpy()
        for variant in VARIANTS:
            x_all = features(df, variant, target)
            preds = [run(df, split, x_all, [target], s)[target] for s in SEEDS]
            summarise(rows, "feature ablation", variant, target, preds, y, named)
            print(f"{target} | {variant:22s} AARE {np.mean([aare(y, p) for p in preds]):6.2f}")

    x_all = features(df, "full", "pb")
    joint = [run(df, split, x_all, ["pb", "bo"], s) for s in SEEDS]
    for target in ("pb", "bo"):
        y = test[target].to_numpy()
        summarise(rows, "joint vs separate", "joint 2-output network", target, [j[target] for j in joint], y, named)
        print(f"{target} | joint network          AARE {np.mean([aare(y, j[target]) for j in joint]):6.2f}")

    out = pd.DataFrame(rows).round(3)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/ablation_studies.csv", index=False)
    print("saved results/ablation_studies.csv")


if __name__ == "__main__":
    main()
