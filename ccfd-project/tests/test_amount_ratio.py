"""
amount_ratio (the base project's FE1 feature) is computed the same way at training time and at serving time.

Training: the base project's apply_feature_engineering adds it to the FE file (p4_01).
Serving:  the base project's prepare_input (p5_04) computes it from the 8 raw fields of each request.
If the two ever differ, the model sees other numbers in production than in training (training/serving skew).
"""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from p5_04_inference_api import TransactionRequest, prepare_input
from utils.feature_engineering_utils import apply_feature_engineering

PROJECT = Path(__file__).resolve().parents[1]
FE_DATA = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
FIELDS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
          "merchant_category", "card_present", "international", "transaction_count_24h"]


def transaction(**changes):
    """The base project's 'suspicious' example from p5_05, with optional changes."""
    record = {"transaction_amount": 2500.00, "transaction_hour": 3, "days_since_last_txn": 0,
              "avg_transaction_amount": 100.00, "merchant_category": "electronics", "card_present": "no",
              "international": "yes", "transaction_count_24h": 7}
    record.update(changes)
    return record


def test_amount_ratio_is_amount_divided_by_average():
    df = prepare_input(TransactionRequest(**transaction()))
    assert list(df.columns) == FIELDS + ["amount_ratio"]
    assert df.loc[0, "amount_ratio"] == 25.0          # 2500 / 100


def test_amounts_are_rounded_to_cents_first():
    df = prepare_input(TransactionRequest(**transaction(transaction_amount=10.006, avg_transaction_amount=3.0)))
    assert df.loc[0, "transaction_amount"] == 10.01
    assert df.loc[0, "amount_ratio"] == 10.01 / 3.0


def test_an_exact_zero_average_is_rejected():
    with pytest.raises(ValidationError):
        TransactionRequest(**transaction(avg_transaction_amount=0))


def test_sub_cent_amounts_keep_base_behavior():
    """The base project checks 'greater than 0' BEFORE it rounds to cents, so 0.004 passes and becomes 0.0.
    Then prepare_input's replace(0, 1) divides by 1: amount_ratio = the amount. The course keeps this as it is
    (parity with p5_04); real data never comes close (smallest average 20.0, smallest amount 5.0)."""
    request = TransactionRequest(**transaction(avg_transaction_amount=0.004))
    assert request.avg_transaction_amount == 0.0
    assert prepare_input(request).loc[0, "amount_ratio"] == 2500.0
    assert TransactionRequest(**transaction(transaction_amount=0.001)).transaction_amount == 0.0


def test_serving_matches_training_on_small_example():
    records = [transaction(), transaction(transaction_amount=50.0, avg_transaction_amount=75.0)]
    training = apply_feature_engineering(pd.DataFrame(records))["amount_ratio"].tolist()
    serving = [prepare_input(TransactionRequest(**r)).loc[0, "amount_ratio"] for r in records]
    assert serving == training


@pytest.mark.skipif(not FE_DATA.exists(), reason="no FE file: run the base project's p1_01 and p4_01 first")
def test_serving_matches_the_fe_file_on_every_row():
    df = pd.read_csv(FE_DATA)
    serving = [prepare_input(TransactionRequest(**r)).loc[0, "amount_ratio"] for r in df[FIELDS].to_dict("records")]
    biggest_gap = (df["amount_ratio"] - pd.Series(serving)).abs().max()
    assert len(serving) == 10000
    assert biggest_gap < 1e-9      # only CSV text rounding (about 1e-14), never a different formula
