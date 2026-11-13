"""
MLOps 03: simulate new data for the fraud project, so we can version it with DVC.

Real data changes over time. This script makes 2 kinds of change on the laptop:

  --mode append    new transactions arrive: the new rows are added at the end of the base project's data file
                   data/credit_card_transactions_latest.csv; the old rows stay exactly as they are
  --mode refresh   the data is made again from scratch (for example, corrected data from a vendor):
                   the output file holds only the new rows

The new rows come from the base project's own code in p1_01_generate_initial_dataset.py (imported, never changed):
  generate_legitimate_transactions, generate_fraudulent_transactions   the base project's 9 columns, in order and
                                                                       The base project's 12 merchant categories
  add_noise    the base project's noise: each amount times a random factor (about 5 percent), each hour
               moved by -1, 0 or +1, and 1 percent of the labels flipped
Before the noise, 2 percent of the new rows are fraud, as in the base project's data (200 of 10,000).
After the noise about 3 percent are fraud: most of the 1 percent flipped labels go from legit to fraud.
The same --seed always makes the same rows, so every data version can be made again, byte for byte.
For example, --mode refresh --rows 10000 --seed 42 makes the base project's data v1 (the same md5).

The data versions of this course (their md5s are in manifest.yaml):
  data v1       python p1_01_generate_initial_dataset.py                                   10,000 rows (the base project)
  data v2       python m03_03_simulate_new_data.py --mode append --rows 5000 --seed 201     15,000 rows
  data v3       python m03_03_simulate_new_data.py --mode append --rows 5000 --seed 202     20,000 rows
  gate holdout  python m03_03_simulate_new_data.py --mode refresh --rows 2000 --seed 301 --output data/gate_holdout.csv

--output is where the result is written. The default is the base project's data file. Append mode always reads the base project's
data file, also when --output names another file. So the gate holdout, with its own --output, never
overwrites a data version.

Append mode stops, and writes nothing, when the rows it would add are already at the end of the base project's data
file (the same command run twice). Each data version needs a new seed.

Run it from the ccfd-project folder (the paths are relative, as in the base project's scripts).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The base project's code, unchanged
from p1_01_generate_initial_dataset import (
    add_noise,
    generate_fraudulent_transactions,
    generate_legitimate_transactions,
)

DATA_FILE = Path("data/credit_card_transactions_latest.csv")   # the base project's data file (data v1 after p1_01)
BASE_COLUMNS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
              "merchant_category", "card_present", "international", "transaction_count_24h", "is_fraud"]
FRAUD_RATE = 0.02     # before the noise, as the base project's p1_01 (200 of 10,000)
MIN_ROWS = 50         # 2 percent of 50 = 1 fraud row
MAX_SEED = 2**32 - 1  # NumPy's random seeds go from 0 to 4,294,967,295
LINE = "=" * 70


def make_rows(rows, seed):
    """Make new transactions with the base project's p1_01 generators and noise. The same rows and seed give the same data."""
    np.random.seed(seed)                       # the base project's generators use NumPy's global random numbers
    n_fraud = int(rows * FRAUD_RATE)
    legit = generate_legitimate_transactions(n=rows - n_fraud)
    fraud = generate_fraudulent_transactions(n=n_fraud)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffle, as p1_01 does
    df = add_noise(df, noise_level=0.05)       # the base project's noise: amounts, hours, 1 percent of the labels flipped
    df["transaction_amount"] = df["transaction_amount"].round(2)
    df["avg_transaction_amount"] = df["avg_transaction_amount"].round(2)
    return df[BASE_COLUMNS]


def describe(df):
    """For example data v2: '15,000 rows (fraud rate 2.97%, 446 fraud)'"""
    n_fraud = int(df["is_fraud"].sum())
    return f"{len(df):,} rows (fraud rate {n_fraud / len(df) * 100:.2f}%, {n_fraud:,} fraud)"


def is_data_file(path):
    """True when path is the base project's data file, also when it is given as a full path or with ../"""
    return path.resolve() == DATA_FILE.resolve()


def replaced_note(output):
    """What happens to output when it is written: a new file, or the rows it had are replaced."""
    if not output.exists():
        return "(a new file)"
    try:
        with output.open(encoding="utf-8") as f:
            old_rows = max(sum(1 for _ in f) - 1, 0)     # the lines without the header (no pandas: any file works)
    except (OSError, UnicodeDecodeError):
        return "(it was not a readable CSV file; it is replaced)"
    return f"(it had {old_rows:,} rows; they are replaced)"


def save(df, output):
    """Write the rows to output as CSV (no index column); make the folder first if needed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def append(rows, seed, output):
    """Add new rows at the end of the base project's data file and write the result to output."""
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Make data v1 first: python p1_01_generate_initial_dataset.py")
        return 1
    # round_trip: every old number is read back exactly, so the old rows are written again byte for byte
    old = pd.read_csv(DATA_FILE, float_precision="round_trip")
    if list(old.columns) != BASE_COLUMNS:
        print(f"ERROR: {DATA_FILE} does not have the base project's 9 columns, in order: {list(old.columns)}")
        return 1
    new = make_rows(rows, seed)
    if len(old) >= rows and old.tail(rows).reset_index(drop=True).equals(new):
        print(f"ERROR: these {rows:,} rows (seed {seed}) are already at the end of {DATA_FILE}; nothing is written.")
        print("       Use a new seed, or go back to the earlier data version first (git checkout and dvc checkout).")
        return 1
    combined = pd.concat([old, new], ignore_index=True)
    note = "" if is_data_file(output) else f" {replaced_note(output)}"
    save(combined, output)

    print(f"  Read:     {DATA_FILE}: {describe(old)}")
    print(f"  New:      {describe(new)}, made with the base project's generators and noise")
    print(f"  Result:   {len(old):,} + {len(new):,} = {describe(combined)}")
    print(f"  Written:  {output}{note}")
    return 0


def refresh(rows, seed, output):
    """Make the data again from scratch: output holds only the new rows."""
    note = replaced_note(output)
    new = make_rows(rows, seed)
    save(new, output)

    print(f"  New:      {describe(new)}, made with the base project's generators and noise")
    print(f"  Written:  {output} {note}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate new data on the base project's schema: append new rows to the base project's data file, or refresh (make it again).")
    parser.add_argument("--mode", choices=["append", "refresh"], required=True,
                        help="append: add rows at the end of the base project's data file; refresh: only the new rows")
    parser.add_argument("--rows", type=int, required=True, help=f"how many new rows (at least {MIN_ROWS})")
    parser.add_argument("--seed", type=int, required=True, help="the random seed: the same seed makes the same rows")
    parser.add_argument("--output", type=Path, default=DATA_FILE,
                        help=f"where to write the result (default: {DATA_FILE}, the base project's data file)")
    args = parser.parse_args(argv)
    if args.rows < MIN_ROWS:
        parser.error(f"--rows must be at least {MIN_ROWS} (2 percent of the rows are fraud)")
    if not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}")
    if args.output.is_dir():
        parser.error(f"--output {args.output} is a folder; give a file name, for example data/gate_holdout.csv")

    print(LINE)
    print(f"m03_03: SIMULATE NEW DATA, mode {args.mode}, {args.rows:,} new rows, seed {args.seed}")
    print(LINE)
    if args.mode == "append":
        code = append(args.rows, args.seed, args.output)
    else:
        code = refresh(args.rows, args.seed, args.output)
    if code == 0 and is_data_file(args.output):
        print(f"  Next:     dvc add {DATA_FILE} (MLOps 03)")
    print(LINE)
    return code


if __name__ == "__main__":
    sys.exit(main())
