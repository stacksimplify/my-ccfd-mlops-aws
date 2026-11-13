"""
Net Benefit, the one business metric behind every gate in this course.

The base project's calculate_business_metrics, with the base project's numbers:
  caught fraud (TP)       +1,475 USD  (1,500 saved minus 25 for the investigation)
  false alarm (FP)           -25 USD  (an investigation for nothing)
  missed fraud (FN)       -1,500 USD
  correct legitimate (TN)      0 USD
"""

from pathlib import Path

import pandas as pd
import pytest

from utils.data_utils import split_data
from utils.metrics_utils import calculate_business_metrics
from utils.model_utils import load_pipeline

PROJECT = Path(__file__).resolve().parents[1]
FE_DATA = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
MODEL_DIR = PROJECT / "models" / "p5_03_production"


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([1, 0], [1, 0], 1475),        # 1 TP + 1 TN
        ([1, 0], [1, 1], 1450),        # 1 TP + 1 FP
        ([1, 0], [0, 0], -1500),       # 1 FN + 1 TN
        ([1, 0, 1, 0], [1, 1, 0, 0], -50),   # one of each: 1475 - 25 - 1500 + 0
    ],
    ids=["tp", "tp-and-fp", "fn", "one-of-each"],
)
def test_net_benefit_per_outcome(y_true, y_pred, expected):
    assert calculate_business_metrics(y_true, y_pred)["net_benefit"] == expected


def test_counts_and_custom_amounts():
    metrics = calculate_business_metrics([1, 1, 0, 0], [1, 0, 1, 0], avg_fraud_amount=1000, investigation_cost=10)
    assert (metrics["fraud_caught"], metrics["fraud_missed"], metrics["false_alarms"], metrics["true_negatives"]) \
        == (1, 1, 1, 1)
    assert metrics["net_benefit"] == 990 - 10 - 1000


@pytest.mark.skipif(not (FE_DATA.exists() and (MODEL_DIR / "model_latest.pkl").exists()),
                    reason="no laptop data or model: run the base project's chain p1_01, p4_01 ... p5_03 first")
def test_tuned_model_reference_is_50550():
    """The base project's reference: p5_03 on its 2,000 test rows gives 50,550 USD."""
    _, X_test, _, y_test, _ = split_data(pd.read_csv(FE_DATA))
    pipeline, _ = load_pipeline(MODEL_DIR.name, save_dir=str(MODEL_DIR.parent))
    assert len(y_test) == 2000
    assert calculate_business_metrics(y_test, pipeline.predict(X_test))["net_benefit"] == 50550
