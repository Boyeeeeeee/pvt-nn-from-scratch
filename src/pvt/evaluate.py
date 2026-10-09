
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvt.baselines import predict_all  # noqa: E402
from pvt.metrics import aare, r2  # noqa: E402
from pvt.model import PVTNet  # noqa: E402
from pvt.train import fit, predict_original, prepare, set_seed  # noqa: E402

CSV, SPLITS = "data/processed/pvt_clean.csv", "data/processed/splits.json"
GRID = {"hidden": [(32,), (64, 64), (128, 128)], "dropout": [0.0, 0.1], "wd": [1e-5, 1e-3]}
GRID_SEEDS, FINAL_SEEDS = range(5), range(10)
UNLABELLED = "numeric_id"
RS_BINS = [(0, 200, "Rs < 200"), (200, 1000, "200-1000"), (1000, 1e9, "Rs > 1000")]


def train_runs(target, cfg, seeds):
    data, xs, ys, df, split = prepare(target, CSV, SPLITS)
    runs = []
    for s in seeds:
        set_seed(s)
        model = PVTNet(4, cfg["hidden"], cfg["dropout"])
        hist = fit(model, data["train"], data["val"], weight_decay=cfg["wd"], seed=s)
        preds = {k: predict_original(model, data[k][0], ys, target) for k in ("val", "test")}
        runs.append((preds, hist))
    return runs, df, split


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    abl_rows, metric_rows, store = [], [], {}

    for target in ("pb", "bo"):
        print(f"\n=== {target.upper()} ===")
        results = []
        for hidden, dropout, wd in itertools.product(GRID["hidden"], GRID["dropout"], GRID["wd"]):
            cfg = {"hidden": hidden, "dropout": dropout, "wd": wd}
            runs, df, split = train_runs(target, cfg, GRID_SEEDS)
            yv = df.loc[split["val"], target].to_numpy()
            yt = df.loc[split["test"], target].to_numpy()
            val = [aare(yv, p["val"]) for p, _ in runs]
            tst = [aare(yt, p["test"]) for p, _ in runs]
            abl_rows.append({"target": target, "hidden": str(hidden), "dropout": dropout, "weight_decay": wd,
                             "val_AARE_mean": np.mean(val), "val_AARE_std": np.std(val),
                             "test_AARE_mean": np.mean(tst), "test_AARE_std": np.std(tst)})
            results.append((np.mean(val), cfg))
            print(f"  {str(hidden):10s} drop={dropout:<4} wd={wd:<7} val AARE {np.mean(val):6.2f}  test AARE {np.mean(tst):6.2f}")
        best_cfg = min(results, key=lambda r: r[0])[1]
        print("  selected on validation:", best_cfg)

        runs, df, split = train_runs(target, best_cfg, FINAL_SEEDS)
        test = df.loc[split["test"]]
        named = (test["source"] != UNLABELLED).to_numpy()
        y = test[target].to_numpy()
        ens = np.mean([p["test"] for p, _ in runs], axis=0)
        per_seed = [p["test"] for p, _ in runs]

        for subset, mask in (("all test", np.ones(len(y), bool)), ("named-source only", named)):
            a = [aare(y[mask], p[mask]) for p in per_seed]
            r = [r2(y[mask], p[mask]) for p in per_seed]
            metric_rows.append({"target": target, "model": "Neural net (mean of 10 seeds)", "subset": subset,
                                "n": int(mask.sum()), "AARE_%": np.mean(a), "AARE_std": np.std(a), "R2": np.mean(r)})
            metric_rows.append({"target": target, "model": "Neural net (10-seed ensemble)", "subset": subset,
                                "n": int(mask.sum()), "AARE_%": aare(y[mask], ens[mask]), "AARE_std": np.nan,
                                "R2": r2(y[mask], ens[mask])})
            for name, p in predict_all(test, target).items():
                metric_rows.append({"target": target, "model": name, "subset": subset, "n": int(mask.sum()),
                                    "AARE_%": aare(y[mask], p[mask]), "AARE_std": np.nan, "R2": r2(y[mask], p[mask])})
        store[target] = {"test": test, "y": y, "ens": ens, "base": predict_all(test, target), "hist": runs[0][1]}

    pd.DataFrame(metric_rows).round(3).to_csv("results/metrics.csv", index=False)
    pd.DataFrame(abl_rows).round(3).to_csv("results/ablations.csv", index=False)
    print("\n", pd.DataFrame(metric_rows).round(2).to_string(index=False))
    make_figures(store)
    print("\nsaved results/metrics.csv, results/ablations.csv, results/figures/*.png")


def make_figures(store):
    # parity plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, target, label in zip(ax, ("pb", "bo"), ("Bubble point pressure (psia)", "Bo (rb/STB)")):
        s = store[target]
        a.scatter(s["y"], s["base"]["Standing"], s=14, alpha=0.5, label="Standing")
        a.scatter(s["y"], s["ens"], s=14, alpha=0.7, label="Neural net")
        lo, hi = min(s["y"].min(), s["ens"].min()), max(s["y"].max(), s["ens"].max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        if target == "pb":
            a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(f"Measured {label}"); a.set_ylabel("Predicted"); a.legend(); a.set_title(f"{target.upper()} - test split")
    fig.tight_layout(); fig.savefig("results/figures/parity_plots.png", dpi=150); plt.close(fig)

    # error by Rs range
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, target in zip(ax, ("pb", "bo")):
        s = store[target]; rs = s["test"]["rs"].to_numpy()
        names, nn_e, st_e = [], [], []
        for lo, hi, lab in RS_BINS:
            m = (rs >= lo) & (rs < hi)
            if m.sum() == 0:
                continue
            names.append(f"{lab}\n(n={m.sum()})")
            nn_e.append(aare(s["y"][m], s["ens"][m])); st_e.append(aare(s["y"][m], s["base"]["Standing"][m]))
        x = np.arange(len(names))
        a.bar(x - 0.2, st_e, 0.4, label="Standing"); a.bar(x + 0.2, nn_e, 0.4, label="Neural net")
        a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel("AARE (%)"); a.set_title(f"{target.upper()} error by GOR range"); a.legend()
    fig.tight_layout(); fig.savefig("results/figures/error_by_rs_range.png", dpi=150); plt.close(fig)

    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, target in zip(ax, ("pb", "bo")):
        h = store[target]["hist"]
        a.plot(h["train_loss"], label="train"); a.plot(h["val_loss"], label="validation")
        a.axvline(h["best_epoch"] - 1, color="k", ls=":", lw=1)
        a.set_xlabel("epoch"); a.set_ylabel("MSE (scaled target)"); a.set_title(f"{target.upper()} training curves"); a.legend()
    fig.tight_layout(); fig.savefig("results/figures/training_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()