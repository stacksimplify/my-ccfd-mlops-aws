"""
MLOps 01 Step-03: mlops_utils/evaluate.py and m01_03_evaluate_endpoint.py.

Both laptop APIs run in this process (FastAPI TestClient, no port): the base project's API answers on POST /inference one
row at a time, the serving API on POST /invocations in one call; both must earn the model's own 50,550 USD. The SageMaker
path runs against a stubbed runtime client (nothing reaches AWS).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

import m01_03_evaluate_endpoint as cli
from mlops_utils import evaluate

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "models" / "p5_03_production"
needs_model = pytest.mark.skipif(
    not ((PROJECT / evaluate.FE_FILE).exists() and (MODEL_DIR / "model_latest.pkl").exists()),
    reason="no laptop data or model: run the base project's chain p1_01, p4_01 ... p5_03 first")


@needs_model
def test_base_project_api_one_call_per_row_earns_its_own_number():
    import p5_04_inference_api as base
    with contextlib.chdir(PROJECT), TestClient(base.app) as client:
        records, labels = evaluate.test_set()
        answers, how, model = evaluate.answers_from_url("http://testserver", records, session=client)
        expected, _ = evaluate.expected_net_benefit(model)
    assert (len(records), sum(labels)) == (2000, 60)
    assert how == "POST /inference, 2,000 calls" and model == "p5_03_production"
    assert evaluate.score(labels, answers) == {"caught": 57, "missed": 3, "false_alarms": 1161, "net_benefit": 50550}
    assert expected == 50550


@needs_model
def test_serving_api_one_call_earns_the_same(monkeypatch):
    import m01_03_serving_api as api
    monkeypatch.setattr(api, "MODEL_DIR", MODEL_DIR)
    with contextlib.chdir(PROJECT), TestClient(api.app) as client:
        records, labels = evaluate.test_set()
        answers, how, model = evaluate.answers_from_url("http://testserver", records, session=client)
    assert how == "POST /invocations, 1 call" and model == "p5_03_production"
    assert evaluate.score(labels, answers)["net_benefit"] == 50550


def test_missing_data_says_what_to_run(tmp_path):
    with pytest.raises(evaluate.EvaluationError, match="run p1_01_generate_initial_dataset.py and p4_01"):
        evaluate.test_set(tmp_path / "nothing.csv")


def test_expected_number_comes_from_the_model_metadata(tmp_path):
    folder = tmp_path / "p3_01_baseline"
    folder.mkdir()
    (folder / "metadata_latest.json").write_text(json.dumps({"metrics": {"net_benefit": "39625"}}))
    assert evaluate.expected_net_benefit("p3_01_baseline", tmp_path) == (39625, str(folder / "metadata_latest.json"))
    assert evaluate.expected_net_benefit("unknown_model", tmp_path)[0] is None


def test_score_refuses_a_count_mismatch():
    with pytest.raises(evaluate.EvaluationError, match="1 answers for 2 transactions"):
        evaluate.score([0, 1], [{"prediction": "FRAUD"}])


def test_endpoint_mode_sends_one_json_list(monkeypatch):
    pytest.importorskip("mlops_utils.endpoints")   # endpoints.py is in the section folders from MLOps 06 on
    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing")):
        monkeypatch.setenv(name, value)
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
    records = [{"transaction_amount": 2500.0}, {"transaction_amount": 50.0}]
    body = json.dumps([{"prediction": "FRAUD", "model_name": "p5_03_production"},
                       {"prediction": "LEGITIMATE", "model_name": "p5_03_production"}]).encode()
    stubber = Stubber(runtime)
    stubber.add_response("invoke_endpoint", {"Body": StreamingBody(io.BytesIO(body), len(body))},
                         expected_params={"EndpointName": "ccfd-rt", "ContentType": "application/json",
                                          "Accept": "application/json", "Body": json.dumps(records)})
    with stubber:
        answers, how, model = evaluate.answers_from_endpoint("ccfd-rt", records, runtime=runtime)
    stubber.assert_no_pending_responses()
    assert [a["prediction"] for a in answers] == ["FRAUD", "LEGITIMATE"]
    assert how == "invoke_endpoint, 1 call (attempts 1)" and model == "p5_03_production"


@pytest.mark.parametrize("expect, code, verdict", [(None, 0, "PASS"), (50550, 0, "PASS"), (99999, 1, "FAIL")])
def test_cli_pass_and_fail(monkeypatch, capsys, expect, code, verdict):
    monkeypatch.setattr(evaluate, "test_set", lambda: ([{}, {}], [1, 0]))
    monkeypatch.setattr(evaluate, "answers_from_url", lambda url, records: (
        [{"prediction": "FRAUD"}, {"prediction": "LEGITIMATE"}], "POST /invocations, 1 call", "p5_03_production"))
    monkeypatch.setattr(evaluate, "score", lambda labels, answers: {
        "caught": 1, "missed": 0, "false_alarms": 0, "net_benefit": 50550})
    monkeypatch.setattr(evaluate, "expected_net_benefit", lambda model: (50550, "models/p5_03_production/metadata_latest.json"))
    argv = ["m01_03_evaluate_endpoint.py", "--url", "http://localhost:8080"] + ([] if expect is None else ["--expect", str(expect)])
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == code
    assert verdict in capsys.readouterr().out
