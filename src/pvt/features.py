"""Input/target transforms shared by training and prediction.

- rs spans ~10 to 6500 scf/STB, so it enters as log10(rs).
- Pb spans two orders of magnitude and is judged by relative error, so it is learned in log10 space.
- Bo (about 1 to 3) is learned on its natural scale.
- Standardizer statistics are fit on the TRAIN split only and saved next to the weights.
"""
import numpy as np

FEATURES = ["rs", "gg", "api", "temp"]


def build_x(df) -> np.ndarray:
    x = df[FEATURES].to_numpy(dtype=float).copy()
    x[:, 0] = np.log10(x[:, 0])
    return x


def to_target(y, target: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log10(y) if target == "pb" else y


def from_target(z, target: str) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 10 ** z if target == "pb" else z


class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, a):
        a = np.asarray(a, dtype=float)
        self.mean = a.mean(axis=0)
        self.std = a.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, a):
        return (np.asarray(a, dtype=float) - self.mean) / self.std

    def inverse(self, a):
        return np.asarray(a, dtype=float) * self.std + self.mean

    def to_dict(self):
        return {"mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]))
