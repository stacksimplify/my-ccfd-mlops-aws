"""
MLOps 11 (BONUS): m11_04_lambda_handler.py with a stubbed SageMaker runtime client.

botocore's Stubber answers every invoke_endpoint call here, so nothing reaches AWS.
A call the test did not queue fails inside the Stubber, so the tests also count the calls.
"""

import importlib
import io
import json

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

BODY = '{"transaction_amount": 2500.0, "transaction_hour": 3}'
ANSWER = '{"prediction": "FRAUD", "probability": 0.998}'
CALL = {"EndpointName": "ccfd-serverless", "ContentType": "application/json",
        "Accept": "application/json", "Body": BODY.encode()}


def load_handler(monkeypatch, endpoint_name=None):
    # Fake credentials and the course region, so making the client never looks for real ones
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "ENDPOINT_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    if endpoint_name:
        monkeypatch.setenv("ENDPOINT_NAME", endpoint_name)
    import m11_04_lambda_handler

    return importlib.reload(m11_04_lambda_handler)   # re-read the settings above


@pytest.fixture
def lam(monkeypatch):
    """The handler with its default endpoint (ENDPOINT_NAME not set)."""
    return load_handler(monkeypatch)


def event(body):
    """A minimal API Gateway HTTP API event (payload format 2.0) for the route POST /inference."""
    return {"version": "2.0", "routeKey": "POST /inference", "body": body, "isBase64Encoded": False}


def answer():
    raw = ANSWER.encode()
    return {"Body": StreamingBody(io.BytesIO(raw), len(raw)), "ContentType": "application/json"}


def call_fails(stubber, account="<ACCOUNT_ID>"):
    """The case seen in a test run: the first call to a new serverless endpoint fails before it reaches the container."""
    stubber.add_client_error("invoke_endpoint", service_error_code="ModelError", http_status_code=424,
                             service_message=f"Received server error (0) from primary in account {account}")


def test_passes_the_json_body_through(lam):
    stubber = Stubber(lam.RUNTIME)
    stubber.add_response("invoke_endpoint", answer(), expected_params=CALL)
    with stubber:
        result = lam.lambda_handler(event(BODY), None)
    stubber.assert_no_pending_responses()
    assert result["statusCode"] == 200
    assert result["headers"] == {"Content-Type": "application/json"}
    assert json.loads(result["body"]) == json.loads(ANSWER)


def test_default_endpoint_is_ccfd_serverless(lam):
    assert lam.ENDPOINT_NAME == "ccfd-serverless"     # MLOps 11; ccfd-prod exists only from MLOps 12 on


def test_endpoint_name_can_be_changed(monkeypatch):
    lam = load_handler(monkeypatch, endpoint_name="ccfd-prod")
    stubber = Stubber(lam.RUNTIME)
    stubber.add_response("invoke_endpoint", answer(), expected_params=dict(CALL, EndpointName="ccfd-prod"))
    with stubber:
        result = lam.lambda_handler(event(BODY), None)
    stubber.assert_no_pending_responses()
    assert result["statusCode"] == 200


def test_first_failure_is_tried_once_more(lam):
    stubber = Stubber(lam.RUNTIME)
    call_fails(stubber)
    stubber.add_response("invoke_endpoint", answer(), expected_params=CALL)
    with stubber:
        result = lam.lambda_handler(event(BODY), None)
    stubber.assert_no_pending_responses()          # exactly 2 calls
    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == json.loads(ANSWER)


def test_second_failure_gives_502_with_the_code_only(lam):
    fake_account = "9" * 12
    stubber = Stubber(lam.RUNTIME)
    call_fails(stubber, fake_account)
    call_fails(stubber, fake_account)
    with stubber:
        result = lam.lambda_handler(event(BODY), None)
    stubber.assert_no_pending_responses()          # exactly 2 calls, never a third
    assert result["statusCode"] == 502
    assert json.loads(result["body"]) == {"detail": "The model endpoint could not be called (ModelError)."}
    assert fake_account not in result["body"]      # AWS's text (it can name the account) never reaches the caller
