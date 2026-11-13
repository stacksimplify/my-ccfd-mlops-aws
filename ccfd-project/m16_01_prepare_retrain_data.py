"""
MLOps 16: the retrain data for the loop (the original monitoring study p7_06, in raw form so the pipeline can check and engineer it).

    python m16_01_prepare_retrain_data.py

Writes, in the base project's raw 9-column format:
  data/retrain/retrain_month1_3.csv   the 8,000 rows p5_03 trained on (the base project's split of data v1) + months 1, 2 and 3 with
                                      their late labels = 23,000 rows (the pipeline's InputDataS3Uri for the loop)
  data/retrain/month_4_holdout.csv    month_4 with labels: the gate data of the loop (GateHoldoutS3Uri); no model
                                      ever trains on it
"""
import hashlib
from pathlib import Path

import pandas as pd

from utils.data_utils import split_data
from utils.feature_engineering_utils import apply_feature_engineering

ORIGINAL = Path("data/credit_card_transactions_latest.csv")     # data v1 (the base project's p1_01, seed 42)
OUT = Path("data/retrain")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build() -> tuple[Path, Path]:
    raw = pd.read_csv(ORIGINAL)
    x_train, _, _, _, _ = split_data(apply_feature_engineering(raw.copy()))   # the base project's split (stratified, 20%, seed 42)
    train_rows = raw.loc[x_train.index]                                       # the same 8,000 rows, raw columns
    months = [pd.read_csv(f"data/production_month_{k}.csv")[raw.columns] for k in (1, 2, 3)]
    retrain = pd.concat([train_rows, *months], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    retrain_file, holdout_file = OUT / "retrain_month1_3.csv", OUT / "month_4_holdout.csv"
    retrain.to_csv(retrain_file, index=False)
    pd.read_csv("data/production_month_4.csv")[raw.columns].to_csv(holdout_file, index=False)
    print(f"{retrain_file}: {len(retrain):,} rows (fraud {retrain['is_fraud'].mean() * 100:.2f}%), md5 {md5(retrain_file)}")
    print(f"{holdout_file}: md5 {md5(holdout_file)}")
    return retrain_file, holdout_file


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
