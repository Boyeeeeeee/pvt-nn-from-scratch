"""Clean the FYP PVT sheet for Project 6 (Pb and Bo prediction).

Usage:  python clean_data.py Data_Sheet__1_.xlsx data/processed/pvt_clean.csv

Keeps only the 11 raw PVT columns, removes non-data rows, fixes impossible
values, drops near-duplicate oils, and adds a rough `source` column used for
group-based train/test splitting.
"""
import re
import sys

import pandas as pd

RAW_COLS = [
    "sn", "pb", "temp", "bo", "sg_oil", "api", "rho_live",
    "gg", "rs", "mu_dead", "mu_sat",
]
NON_DATA_LABELS = {"count", "min", "max", "avg", "mean", "std"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).iloc[:, :11]
    df.columns = RAW_COLS  # PRES column is the saturation pressure (renamed pb)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(df)}

    # 1. blank rows, summary rows (count/min/max/avg), synthetic rows
    df = df[df["sn"].notna()].copy()
    log["after_blank_removed"] = len(df)
    sn = df["sn"].astype(str).str.strip()
    is_summary = sn.str.lower().isin(NON_DATA_LABELS)
    is_synthetic = sn.str.lower().str.startswith("generated")
    log["summary_rows_dropped"] = int(is_summary.sum())
    log["synthetic_rows_dropped"] = int(is_synthetic.sum())
    df = df[~(is_summary | is_synthetic)].copy()

    # 2. numeric coercion, then required inputs/targets must be present
    for c in RAW_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    required = ["pb", "temp", "bo", "api", "gg", "rs"]
    before = len(df)
    df = df.dropna(subset=required)
    log["missing_required_dropped"] = before - len(df)

    # 3. physically impossible values (gas gravity of 0 means missing)
    before = len(df)
    df = df[(df["gg"] > 0) & (df["rs"] > 0) & (df["pb"] > 0)]
    log["impossible_values_dropped"] = before - len(df)

    # 4. near-duplicate oils (same oil reported by two sources)
    key = (
        df["api"].round(1).astype(str) + "|" + df["temp"].round(0).astype(str)
        + "|" + df["gg"].round(3).astype(str) + "|" + df["pb"].round(0).astype(str)
    )
    before = len(df)
    df = df[~key.duplicated(keep="first")]
    log["near_duplicates_dropped"] = before - len(df)

    # 5. rough source label from the S/N prefix, used for group splitting
    def source(s: str) -> str:
        m = re.match(r"^([A-Za-z]+)", str(s).strip())
        return m.group(1) if m else "numeric_id"

    df["source"] = df["sn"].map(source)
    log["clean_rows"] = len(df)
    return df.reset_index(drop=True), log


def report_outliers(df: pd.DataFrame) -> None:
    print("\nValues to inspect (kept, not removed):")
    print(f"  rs > 3000 scf/STB : {(df['rs'] > 3000).sum()}")
    print(f"  bo > 3            : {(df['bo'] > 3).sum()}")
    print(f"  api < 12          : {(df['api'] < 12).sum()}")
    print(f"  pb > 8000 psi     : {(df['pb'] > 8000).sum()}")
    print("\nRows per source label (check these before trusting the group split):")
    print(df["source"].value_counts().head(15).to_string())


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Data_Sheet__1_.xlsx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "pvt_clean.csv"
    clean_df, log = clean(load(src))
    for k, v in log.items():
        print(f"{k:28s}{v}")
    report_outliers(clean_df)
    clean_df.to_csv(dst, index=False)
    print(f"\nSaved {len(clean_df)} rows -> {dst}")
