"""Group train/val/test split by `source`, fixed seed, saved to data/processed/splits.json.

TODO: sklearn GroupShuffleSplit or GroupKFold. No source may appear in both train and test.
Baselines and the network must evaluate on the same saved test ids.
"""
