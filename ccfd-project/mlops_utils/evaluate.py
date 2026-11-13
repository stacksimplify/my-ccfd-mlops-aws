"""
mlops_utils/evaluate.py: evaluate ANY endpoint on the base project's 2,000 test transactions.

The MLOps question it answers: does this deployed model earn what training promised (training-serving skew)?

  test_set()                 the base project's split (stratified, 20%, seed 42) of the FE1 data: 2,000 rows, 60 frauds;
                             only the 8 fields a client sends (each API computes amount_ratio itself)
  answers_from_url(url, ..)  a laptop API or a container: GET /ping 200 -> the SageMaker contract, ONE POST
                             /invocations with all rows; no /ping (the base project's API) -> POST /inference per row
  answers_from_endpoint(..)  a SageMaker endpoint: one invoke_endpoint call with all rows (exactly one retry, like every
                             call in this course); needs mlops_utils/endpoints.py (the section folders from MLOps 07 on)
  expected_net_benefit(..)   the number the model saved at training time: models/<model_name>/metadata_latest.json
  score(labels, answers)     caught, missed, false alarms and Net Benefit with the base project's money rule

No MLflow and no file writes here; the command line is m01_03_evaluate_endpoint.py.
"""

import contextlib
import io
import json
from pathlib import Path

import httpx
import pandas as pd

from utils.data_utils import split_data
from utils.metrics_utils import calculate_business_metrics

FE_FILE = Path("data/credit_card_transactions_fe1_amount_ratio.csv")   # the base project's FE1 data (p4_01 output)
FIELDS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
          "merchant_category", "card_present", "international", "transaction_count_24h"]
TIMEOUT_SECONDS = 300


class EvaluationError(Exception):
    """Anything that stops an evaluation, with a message a student can act on."""


def test_set(fe_file: Path = FE_FILE) -> tuple[list[dict], list[int]]:
    """The 2,000 test transactions (8 fields each) and their true labels (1 = fraud)."""
    if not Path(fe_file).exists():
        raise EvaluationError(f"no {fe_file}: run p1_01_generate_initial_dataset.py and p4_01_feature_engineering_fe1.py "
                              "first (Section 00-A Step-06)")
    with contextlib.redirect_stdout(io.StringIO()):   # the base project's split_data prints emoji; keep our output ASCII
        _, x_test, _, y_test, _ = split_data(pd.read_csv(fe_file))
    return x_test[FIELDS].to_dict("records"), [int(label) for label in y_test]


def answers_from_url(url: str, records: list[dict], session=None) -> tuple[list[dict], str, str]:
    """Returns (answers, how it was called, model_name from /health)."""
    http = session if session is not None else httpx.Client()
    base = url.rstrip("/")
    try:
        health = http.get(f"{base}/health", timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as err:
        raise EvaluationError(f"nothing answers at {base} ({type(err).__name__}): start the API first") from None
    if health.status_code != 200 or not health.json().get("model_loaded"):
        raise EvaluationError(f"{base}/health says the model is not loaded: {health.text[:200]}")
    model_name = health.json().get("model_name", "unknown")

    if http.get(f"{base}/ping", timeout=TIMEOUT_SECONDS).status_code == 200:   # the SageMaker contract
        response = http.post(f"{base}/invocations", json=records, timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise EvaluationError(f"POST /invocations: HTTP {response.status_code}: {response.text[:200]}")
        return response.json(), "POST /invocations, 1 call", model_name

    answers = []                                                               # the base project's API
    for record in records:
        response = http.post(f"{base}/inference", json=record, timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise EvaluationError(f"POST /inference: HTTP {response.status_code}: {response.text[:200]}")
        answers.append(response.json())
    return answers, f"POST /inference, {len(records):,} calls", model_name


def answers_from_endpoint(endpoint: str, records: list[dict], runtime=None) -> tuple[list[dict], str, str]:
    """Returns (answers, how it was called, model_name from the answers)."""
    try:
        from mlops_utils import endpoints
    except ImportError:
        raise EvaluationError("--endpoint needs mlops_utils/endpoints.py: run it from an AWS section folder "
                              "(MLOps 07 on)") from None
    result = endpoints.invoke(endpoint, records, runtime=runtime)
    answers = json.loads(result["body"])
    return answers, f"invoke_endpoint, 1 call (attempts {result['attempts']})", answers[0].get("model_name", "unknown")


def expected_net_benefit(model_name: str, models_dir: Path = Path("models")) -> tuple[int | None, str]:
    """(the Net Benefit the model saved at training time, where it came from), or (None, reason)."""
    metadata = Path(models_dir) / model_name / "metadata_latest.json"
    if not metadata.exists():
        return None, f"no {metadata}: give --expect"
    value = json.loads(metadata.read_text()).get("metrics", {}).get("net_benefit")
    if value is None:
        return None, f"{metadata} has no metrics.net_benefit: give --expect"
    return int(value), str(metadata)


def score(labels: list[int], answers: list[dict]) -> dict:
    """The base project's money rule on the answers: caught, missed, false alarms, Net Benefit."""
    if len(answers) != len(labels):
        raise EvaluationError(f"{len(answers)} answers for {len(labels)} transactions")
    predicted = [1 if answer["prediction"] == "FRAUD" else 0 for answer in answers]
    metrics = calculate_business_metrics(labels, predicted)
    return {"caught": int(metrics["fraud_caught"]), "missed": int(metrics["fraud_missed"]),
            "false_alarms": int(metrics["false_alarms"]), "net_benefit": int(metrics["net_benefit"])}
