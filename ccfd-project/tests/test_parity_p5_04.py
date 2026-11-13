"""
Parity test: the base project's laptop API p5_04 and the course serving API give the same answers.

The data: the base project's 2,000 test rows (split_data on data/credit_card_transactions_fe1_amount_ratio.csv, seed 42).
The p5_04 side: the base project's own p5_04_inference_api.py (unchanged), one POST /inference per transaction, as p5_05 does.
The serving side: all 2,000 transactions in one request, as a JSON list (options in tests/conftest.py):
  - default: m01_03_serving_api in this Python process (no server needed)
  - --url http://localhost:8080 (or PARITY_URL): a running server or container; the test posts to <url>/invocations
        python -m pytest -s tests/test_parity_p5_04.py --url http://localhost:8080
  - --endpoint ccfd-rt (or PARITY_ENDPOINT): a SageMaker endpoint, with boto3 invoke_endpoint (paid sessions only)

It checks: 2000/2000 labels equal, the largest probability difference, and the Net Benefit of both sides
with the base project's calculate_business_metrics (the base project's reference: 50,550 USD). It prints one line that names
the serving side it tested:
    Parity: 2000/2000 labels equal, max |prob diff| = <x>, Net Benefit p5_04 = <a>, serving (<where>) = <b>
The serving API scores the 2,000 rows in one predict_proba call and p5_04 one row at a time, so a few
probabilities differ in the last binary digit (2.22e-16 measured on the M5); labels and Net Benefit are equal.
"""

import contextlib
import io
import json
import os
from pathlib import Path

import boto3
import pandas as pd
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from utils.data_utils import split_data
from utils.metrics_utils import calculate_business_metrics

PROJECT = Path(__file__).resolve().parents[1]
FE_DATA = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
MODEL_DIR = PROJECT / "models" / "p5_03_production"

# The 8 raw fields a client sends (amount_ratio is computed by the API)
FIELDS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
          "merchant_category", "card_present", "international", "transaction_count_24h"]
REFERENCE_NET_BENEFIT = 50550   # the base project's p5_03 on its test rows
MAX_PROB_DIFF = 1e-6            # one call for all rows vs one call per row, or another CPU: last digits only


def p5_04_answers(records):
    """The base project's p5_04 API, unchanged, in this process: one POST /inference per transaction."""
    import p5_04_inference_api as base

    # The base project loads models/p5_03_production relative to the current folder, so we run it from ccfd-project
    with contextlib.chdir(PROJECT), TestClient(base.app) as client:
        answers = []
        for record in records:
            response = client.post("/inference", json=record)
            assert response.status_code == 200, response.text
            answers.append(response.json())
    return answers


def endpoint_answers(endpoint, records, runtime=None):
    """A SageMaker endpoint: one invoke_endpoint call with all rows as a JSON list (about 0.5 MB for 2,000
    rows; the request limit is 6 MB). boto3 signs the call with your AWS profile."""
    if runtime is None:
        runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                                       Accept="application/json", Body=json.dumps(records))
    return json.loads(response["Body"].read())


def serving_answers(records, url=None, endpoint=None):
    """The course serving API: all transactions in one request. Returns (answers, where it ran)."""
    if endpoint:
        return endpoint_answers(endpoint, records), f"endpoint {endpoint}"
    if url:
        import httpx

        with httpx.Client(base_url=url, timeout=300) as client:
            health = client.get("/health")       # which server answers, and from which model folder
            assert health.status_code == 200 and health.json()["model_loaded"], health.text
            response = client.post("/invocations", json=records)
        where = f"{url}, model_dir {health.json()['model_dir']}"
    else:
        import m01_03_serving_api as api

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(api, "MODEL_DIR", MODEL_DIR)   # always the laptop model
            with TestClient(api.app) as client:
                response = client.post("/invocations", json=records)
        where = "in-process"
    assert response.status_code == 200, response.text
    return response.json(), where


def net_benefit(y_true, answers):
    y_pred = [1 if answer["prediction"] == "FRAUD" else 0 for answer in answers]
    return int(calculate_business_metrics(y_true, y_pred)["net_benefit"])


def test_parity_p5_04_vs_serving(request, capsys):
    if not (FE_DATA.exists() and (MODEL_DIR / "model_latest.pkl").exists()):
        pytest.skip("no laptop data or model: run the base project's chain p1_01, p4_01 ... p5_03 first")
    df = pd.read_csv(FE_DATA)
    _, X_test, _, y_test, _ = split_data(df)          # the base project's split: stratified, 20%, seed 42
    records = X_test[FIELDS].to_dict("records")
    assert len(records) == 2000

    laptop = p5_04_answers(records)
    serving, where = serving_answers(records, request.config.getoption("--url"),
                                     request.config.getoption("--endpoint"))
    assert len(serving) == len(laptop)

    labels_equal = sum(a["prediction"] == b["prediction"] for a, b in zip(laptop, serving, strict=True))
    max_diff = max(abs(a["probability"] - b["probability"]) for a, b in zip(laptop, serving, strict=True))
    nb_p5_04 = net_benefit(y_test, laptop)
    nb_serving = net_benefit(y_test, serving)

    diff_text = "0.0" if max_diff == 0 else f"{max_diff:.2e}"
    line = (f"Parity: {labels_equal}/{len(laptop)} labels equal, max |prob diff| = {diff_text}, "
            f"Net Benefit p5_04 = {nb_p5_04:,}, serving ({where}) = {nb_serving:,}")
    with capsys.disabled():                           # show the line even with pytest -q
        print("\n" + line)

    assert labels_equal == len(laptop) == 2000
    assert max_diff < MAX_PROB_DIFF
    assert nb_p5_04 == REFERENCE_NET_BENEFIT
    assert nb_serving == REFERENCE_NET_BENEFIT


def test_endpoint_mode_sends_one_json_list(monkeypatch):
    """The --endpoint call shape, with a stubbed SageMaker runtime client (nothing reaches AWS)."""
    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing")):
        monkeypatch.setenv(name, value)
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
    records = [{"transaction_amount": 2500.0}, {"transaction_amount": 50.0}]
    answer = json.dumps([{"prediction": "FRAUD"}, {"prediction": "LEGITIMATE"}]).encode()
    stubber = Stubber(runtime)
    stubber.add_response(
        "invoke_endpoint", {"Body": StreamingBody(io.BytesIO(answer), len(answer))},
        expected_params={"EndpointName": "ccfd-rt", "ContentType": "application/json",
                         "Accept": "application/json", "Body": json.dumps(records)},
    )
    with stubber:
        answers = endpoint_answers("ccfd-rt", records, runtime)
    stubber.assert_no_pending_responses()
    assert [a["prediction"] for a in answers] == ["FRAUD", "LEGITIMATE"]
