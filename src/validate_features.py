"""
Checks that stmacs_features.FeatureBuilder reproduces the 54 CORE features
stored in the real system 455 training data.

It streams the raw sensor columns (as a live twin would deliver them) through
the builder, then compares every feature it produces with the value stored in
the CSV. Rows in the CSV where history is broken (data gaps) are expected to
differ slightly, because the original features were built by row-shifting the
full dataset, which jumps across gaps.

Usage
-----
  python validate_features.py                     uses the default 60-minute
                                                  training CSV
  python validate_features.py --csv PATH --rows 20000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stmacs_features import FeatureBuilder, InsufficientHistory

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--rows", type=int, default=20000,
                    help="how many rows to stream (more = slower)")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
csv_path = (Path(args.csv) if args.csv else
            BASE_DIR / "data" / "processed" / "ml" / "system455_60min_train.csv")

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.iloc[:args.rows].reset_index(drop=True)
print(f"Streaming {len(df):,} rows from {csv_path.name}")

fb = FeatureBuilder()
produced = []
row_index = []

for i, r in df.iterrows():
    fb.update(r["timestamp"], r["buffer_top"], r["buffer_centre"],
              r["buffer_bottom"], r["collector_temp"], r["solar_flow"],
              r["solar_pump"])
    try:
        produced.append(fb.features())
        row_index.append(i)
    except InsufficientHistory:
        pass

out = pd.DataFrame(produced, index=row_index)
print(f"Features produced for {len(out):,} of {len(df):,} rows "
      f"(the rest lacked history)")
print(f"Feature count: {out.shape[1]} (expected 54)")

missing_cols = [c for c in out.columns if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Columns not in CSV: {missing_cols}")

print(f"\n{'Feature':<40}{'exact-match %':>14}{'max abs diff':>14}")
worst = []
for col in out.columns:
    a = out[col].to_numpy()
    b = df.loc[out.index, col].to_numpy()
    ok = np.isclose(a, b, atol=1e-6)
    worst.append((ok.mean(), col, float(np.max(np.abs(a - b)))))

worst.sort()
for frac, col, mx in worst[:12]:
    print(f"{col:<40}{frac * 100:>14.2f}{mx:>14.3f}")
print("  ... (12 lowest-matching features shown)")

overall = np.mean([w[0] for w in worst])
print(f"\nAverage exact-match across all 54 features: {overall * 100:.2f}%")
print("Time, stratification and collector-difference features should be 100%.")
print("Lag and delta features may fall a little short (typically under 0.5%)")
print("where the CSV spans a gap in the logged data.")