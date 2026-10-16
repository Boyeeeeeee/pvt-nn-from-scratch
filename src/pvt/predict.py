"""Load the trained Pb and Bo networks and predict from raw inputs.

    from pvt.predict import Predictor
    Predictor().predict(rs=450, gg=0.75, api=32, temp=210)
    -> {"pb_psia": ..., "bo_rb_per_stb": ..., "warnings": [...]}

Inputs: rs = solution GOR (scf/STB), gg = gas specific gravity (air = 1), api = oil API gravity,
temp = reservoir temperature (deg F). Valid for saturated oils only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pvt.features import FEATURES, Standardizer, build_x, from_target
from pvt.model import PVTNet

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
LOW_GOR_LIMIT = 200.0


class Predictor:
    def __init__(self, models_dir=None):
        d = Path(models_dir) if models_dir else MODELS_DIR
        self.nets, self.xs, self.ys, self.ranges = {}, {}, {}, {}
        for target in ("pb", "bo"):
            ckpt = torch.load(d / f"{target}_model.pt", map_location="cpu", weights_only=True)
            net = PVTNet(4, tuple(ckpt["hidden"]), ckpt["dropout"])
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            sc = json.loads((d / f"{target}_scalers.json").read_text())
            self.nets[target] = net
            self.xs[target] = Standardizer.from_dict(sc["x"])
            self.ys[target] = Standardizer.from_dict(sc["y"])
            self.ranges = sc.get("ranges", self.ranges)

    def _one(self, target, df):
        x = torch.tensor(self.xs[target].transform(build_x(df)), dtype=torch.float32)
        with torch.no_grad():
            z = self.ys[target].inverse(self.nets[target](x).numpy().reshape(-1, 1)).ravel()
        return float(from_target(z, target)[0])

    def warnings(self, rs, gg, api, temp):
        out, vals = [], dict(rs=rs, gg=gg, api=api, temp=temp)
        for name in FEATURES:
            if name in self.ranges:
                lo, hi = self.ranges[name]
                if not lo <= vals[name] <= hi:
                    out.append(f"{name}={vals[name]} is outside the training range [{lo:g}, {hi:g}]; "
                               "prediction is an extrapolation.")
        if rs < LOW_GOR_LIMIT:
            out.append(f"rs < {LOW_GOR_LIMIT:g} scf/STB: Pb predictions are less reliable in this range "
                       "(test error was about 40% here).")
        return out

    def predict(self, rs: float, gg: float, api: float, temp: float) -> dict:
        df = pd.DataFrame([{"rs": rs, "gg": gg, "api": api, "temp": temp}])
        return {
            "pb_psia": self._one("pb", df),
            "bo_rb_per_stb": self._one("bo", df),
            "warnings": self.warnings(rs, gg, api, temp),
        }
