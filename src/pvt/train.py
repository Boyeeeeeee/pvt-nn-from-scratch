"""Hand-written training loop for one target.

Usage (repo root):  python src/pvt/train.py --target pb
                    python src/pvt/train.py --target bo

Everything the loop does is visible here: mini-batching, forward pass, loss, backward(),
optimizer step, validation each epoch, early stopping on validation loss, and checkpointing
the best weights. No Trainer/Lightning wrappers.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvt.features import FEATURES, Standardizer, build_x, from_target, to_target  # noqa: E402
from pvt.metrics import aare, r2  # noqa: E402
from pvt.model import PVTNet  # noqa: E402
from pvt.splits import load_split  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare(target: str, csv: str, splits_json: str):
    """Return scaled tensors for each split plus the fitted scalers (fit on train only)."""
    df = pd.read_csv(csv)
    split = load_split(splits_json)
    x_all, z_all = build_x(df), to_target(df[target], target)
    xs = Standardizer().fit(x_all[split["train"]])
    ys = Standardizer().fit(z_all[split["train"]].reshape(-1, 1))
    data = {}
    for name, idx in split.items():
        x = torch.tensor(xs.transform(x_all[idx]), dtype=torch.float32)
        y = torch.tensor(ys.transform(z_all[idx].reshape(-1, 1)).ravel(), dtype=torch.float32)
        data[name] = (x, y)
    return data, xs, ys, df, split


def fit(model, train, val, lr=3e-3, weight_decay=1e-5, batch_size=32,
        max_epochs=500, patience=50, seed=42):
    set_seed(seed)
    (xtr, ytr), (xva, yva) = train, val
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        running = 0.0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(xtr[idx])              # forward
            loss = loss_fn(pred, ytr[idx])      # loss
            opt.zero_grad()
            loss.backward()                     # backward
            opt.step()                          # update
            running += loss.item() * len(idx)
        train_loss = running / len(xtr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xva), yva).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"], history["best_val_loss"] = best_epoch, best_val
    return history


def predict_original(model, x_scaled, ys: Standardizer, target: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = ys.inverse(model(x_scaled).numpy().reshape(-1, 1)).ravel()
    return from_target(z, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["pb", "bo"], required=True)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default="data/processed/pvt_clean.csv")
    p.add_argument("--splits", default="data/processed/splits.json")
    p.add_argument("--out-dir", default="models")
    a = p.parse_args()

    data, xs, ys, df, split = prepare(a.target, a.csv, a.splits)
    set_seed(a.seed)
    model = PVTNet(in_dim=4, hidden=tuple(a.hidden), dropout=a.dropout)
    hist = fit(model, data["train"], data["val"], a.lr, a.weight_decay,
               a.batch_size, a.epochs, a.patience, a.seed)

    print(f"target={a.target}  best epoch={hist['best_epoch']}  epochs run={len(hist['val_loss'])}")
    for name in ("train", "val", "test"):
        y_true = df.loc[split[name], a.target].to_numpy()
        y_hat = predict_original(model, data[name][0], ys, a.target)
        print(f"  {name:5s} n={len(y_true):3d}  AARE {aare(y_true, y_hat):6.2f}%  R2 {r2(y_true, y_hat):.4f}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": a.hidden, "dropout": a.dropout},
               out / f"{a.target}_model.pt")
    with open(out / f"{a.target}_scalers.json", "w") as f:
        ranges = {c: [float(df.loc[split["train"], c].min()), float(df.loc[split["train"], c].max())]
                  for c in FEATURES}
        json.dump({"x": xs.to_dict(), "y": ys.to_dict(), "target": a.target, "ranges": ranges}, f)
    Path("results").mkdir(exist_ok=True)
    with open(f"results/history_{a.target}.json", "w") as f:
        json.dump(hist, f)
    print(f"saved -> {out}/{a.target}_model.pt, {a.target}_scalers.json")


if __name__ == "__main__":
    main()
