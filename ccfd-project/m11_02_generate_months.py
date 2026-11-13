"""
MLOps 11 (also MLOps 15 and 16): 4 simulated production months that drift away from the training data.

A port of p7_02_generate_production_data.py, the month generator of the original monitoring study: the same settings,
the same seeds and the same order of random numbers, so the 4 files are byte for byte the base project's files (the same md5).
The data is made up by this script; there is no real bank traffic here.

  month    seed  spending  international  stealth share of fraud  used for
  month_1  101   x 1.00     8 percent       0 percent             a control: the world of the training data
  month_2  102   x 1.15    15 percent      55 percent             mild drift
  month_3  103   x 1.35    24 percent      90 percent             strong drift
  month_4  104   x 1.40    25 percent      90 percent             the next month, the same world as month_3;
                                                                  kept aside: no model ever trains on it

Each month has 5,000 rows and the fraud rate is fixed at 2 percent (100 rows). Only the kind of fraud changes:
  classic fraud   big amounts, international, card not present, at night (what the model learned)
  stealth fraud   domestic, card present, daytime, common merchants, about 2 to 3 times the customer's
                  normal spend (amount_ratio about 2 to 3); a small slice of all rows, so data drift (PSI)
                  hardly sees it; only performance monitoring with labels does
Honest customers drift too: they spend more, go abroad more and shop online more.

Each file has the base project's 9 columns, in order (the 8 request fields + is_fraud). The months get the base project's light
noise on amounts and hours, but no label flips (as in p7_02).

Used in: MLOps 11 (Batch Transform on month_1), MLOps 15 (replay and drift on months 1 to 4),
MLOps 16 (month_4 is the gate holdout of the retrain).

Run it from the ccfd-project folder:
    python m11_02_generate_months.py
Output: data/production_month_1.csv to data/production_month_4.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 5000
FRAUD_RATE = 0.02
MONTHS = [   # the settings of the original month generator (p7_02), unchanged
    {"name": "month_1", "amount_inflation": 1.00, "intl_share": 0.08, "stealth_frac": 0.00, "seed": 101},
    {"name": "month_2", "amount_inflation": 1.15, "intl_share": 0.15, "stealth_frac": 0.55, "seed": 102},
    {"name": "month_3", "amount_inflation": 1.35, "intl_share": 0.24, "stealth_frac": 0.90, "seed": 103},
    {"name": "month_4", "amount_inflation": 1.40, "intl_share": 0.25, "stealth_frac": 0.90, "seed": 104},
]
MERCHANTS = ["grocery", "gas_station", "restaurant", "retail", "online_shopping",
             "pharmacy", "entertainment", "utilities", "travel", "electronics",
             "furniture", "jewelry"]   # the base project's 12 categories
LINE = "=" * 70


def generate_legit(n, amount_inflation, intl_share, rng):
    """Honest transactions: the base project's p1_01 patterns, with higher spending and more international travel."""
    hour_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                           0.03, 0.05, 0.06, 0.08, 0.08, 0.08,
                           0.08, 0.07, 0.06, 0.06, 0.06, 0.07,
                           0.08, 0.07, 0.05, 0.03, 0.02, 0.02])
    hour_probs = hour_probs / hour_probs.sum()

    n_low = int(n * 0.85)
    n_high = n - n_low
    low = rng.gamma(shape=2, scale=80, size=n_low).clip(5, 1000)
    high = rng.gamma(shape=3, scale=300, size=n_high).clip(500, 5000)
    amounts = np.concatenate([low, high]) * amount_inflation
    rng.shuffle(amounts)

    # As spending goes up, more shopping goes online: online_shopping up, grocery down by the same amount
    online_boost = min(0.10 * (amount_inflation - 1.0) / 0.35, 0.10)
    p = np.array([0.20, 0.11, 0.18, 0.12, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.01])
    p[4] += online_boost
    p[0] -= online_boost
    p = np.clip(p, 0.001, None)
    p = p / p.sum()

    return pd.DataFrame({
        "transaction_amount": amounts,
        "transaction_hour": rng.choice(range(24), size=n, p=hour_probs),
        "days_since_last_txn": rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 14, 30], size=n,
                                          p=[0.05, 0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]),
        "avg_transaction_amount": (rng.gamma(shape=2, scale=90, size=n).clip(20, 1500)) * amount_inflation,
        "merchant_category": rng.choice(MERCHANTS, size=n, p=p),
        "card_present": rng.choice(["yes", "no"], size=n, p=[0.70, 0.30]),
        "international": rng.choice(["no", "yes"], size=n, p=[1 - intl_share, intl_share]),
        "transaction_count_24h": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.50, 0.30, 0.12, 0.05, 0.03]),
        "is_fraud": [0] * n,
    })


def generate_classic_fraud(n, rng):
    """The old, loud fraud: the base project's p1_01 fraud patterns (big, international, card not present)."""
    if n <= 0:
        return None
    hour_probs = np.array([0.08, 0.08, 0.07, 0.06, 0.05, 0.04,
                           0.03, 0.03, 0.04, 0.05, 0.05, 0.05,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.04, 0.04, 0.03, 0.03, 0.03])
    hour_probs = hour_probs / hour_probs.sum()
    n_test = int(n * 0.20)
    n_large = n - n_test
    amounts = np.concatenate([rng.uniform(5, 100, size=n_test),
                              rng.uniform(500, 3000, size=n_large)])
    rng.shuffle(amounts)
    return pd.DataFrame({
        "transaction_amount": amounts,
        "transaction_hour": rng.choice(range(24), size=n, p=hour_probs),
        "days_since_last_txn": rng.choice([0, 1, 2, 3, 4, 7], size=n, p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05]),
        "avg_transaction_amount": rng.gamma(shape=2, scale=70, size=n).clip(20, 500),
        "merchant_category": rng.choice(
            ["electronics", "jewelry", "online_shopping", "travel", "retail",
             "gas_station", "grocery", "restaurant", "entertainment"], size=n,
            p=[0.25, 0.15, 0.20, 0.08, 0.12, 0.08, 0.05, 0.05, 0.02]),
        "card_present": rng.choice(["no", "yes"], size=n, p=[0.70, 0.30]),
        "international": rng.choice(["yes", "no"], size=n, p=[0.30, 0.70]),
        "transaction_count_24h": rng.choice([2, 3, 4, 5, 6, 7, 8], size=n,
                                            p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05]),
        "is_fraud": [1] * n,
    })


def generate_stealth_fraud(n, rng):
    """The new fraud: it looks ordinary; the only sign is an amount about 2 to 3 times the normal spend."""
    if n <= 0:
        return None
    avg_amount = rng.gamma(shape=2, scale=90, size=n).clip(20, 1500)
    amounts = avg_amount * rng.uniform(2.0, 3.2, size=n)
    return pd.DataFrame({
        "transaction_amount": amounts,
        "transaction_hour": rng.integers(9, 19, size=n),          # daytime
        "days_since_last_txn": rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2]),
        "avg_transaction_amount": avg_amount,
        "merchant_category": rng.choice(
            ["online_shopping", "retail", "electronics", "grocery", "restaurant"],
            size=n, p=[0.40, 0.25, 0.15, 0.10, 0.10]),
        "card_present": rng.choice(["yes", "no"], size=n, p=[0.80, 0.20]),
        "international": rng.choice(["no", "yes"], size=n, p=[0.90, 0.10]),
        "transaction_count_24h": rng.choice([1, 2, 3, 4], size=n, p=[0.40, 0.35, 0.15, 0.10]),
        "is_fraud": [1] * n,
    })


def light_noise(df, rng, noise_level=0.05):
    """The base project's p1_01 noise on amounts and hours, without the label flips."""
    df = df.copy()
    noise = rng.normal(1.0, noise_level, size=len(df))
    df["transaction_amount"] = (df["transaction_amount"] * noise).clip(5, 8000)
    hour_noise = rng.integers(-1, 2, size=len(df))
    df["transaction_hour"] = (df["transaction_hour"] + hour_noise).clip(0, 23)
    return df


def make_month(amount_inflation, intl_share, stealth_frac, seed, n=ROWS, fraud_rate=FRAUD_RATE):
    """One month of transactions. The fraud rate is fixed; only the mix of fraud changes."""
    rng = np.random.default_rng(seed)

    n_fraud = round(n * fraud_rate)
    n_legit = n - n_fraud
    n_stealth = round(n_fraud * stealth_frac)
    n_classic = n_fraud - n_stealth

    parts = [generate_legit(n_legit, amount_inflation, intl_share, rng)]
    classic = generate_classic_fraud(n_classic, rng)
    stealth = generate_stealth_fraud(n_stealth, rng)
    if classic is not None:
        parts.append(classic)
    if stealth is not None:
        parts.append(stealth)

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = light_noise(df, rng)
    df["transaction_amount"] = df["transaction_amount"].round(2)
    df["avg_transaction_amount"] = df["avg_transaction_amount"].round(2)
    return df, n_classic, n_stealth


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the 4 simulated production months.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="the folder for production_month_1.csv to _4.csv (default: data)")
    args = parser.parse_args(argv)

    print(LINE)
    print("m11_02: 4 SIMULATED PRODUCTION MONTHS, fraud fixed at 2 percent")
    print(LINE)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        df, n_classic, n_stealth = make_month(month["amount_inflation"], month["intl_share"],
                                              month["stealth_frac"], month["seed"])
        path = args.data_dir / f"production_{month['name']}.csv"
        df.to_csv(path, index=False)
        # One fact per line, each with its label
        print(f"  {month['name']}: {len(df):,} rows -> {path}")
        print(f"    fraud:          {df['is_fraud'].mean() * 100:.1f}% ({n_classic} classic + {n_stealth} stealth)")
        print(f"    average amount: ${df['transaction_amount'].mean():.2f}")
        print(f"    international:  {month['intl_share'] * 100:.1f}%")
    print(LINE)
    print("month_4 is kept aside: no model trains on it (MLOps 16 uses it as the gate holdout).")
    print(LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
