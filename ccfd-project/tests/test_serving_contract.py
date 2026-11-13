"""
MLOps 01: the serving contract of m01_03_serving_api.py.

GET /ping, POST /invocations with one JSON object, a JSON list and JSON Lines, HTTP 400 on bad input,
HTTP 415 on other content types, model_name and threshold 0.5 in every row, the base project's 3 test transactions
(3/3 FRAUD), and the laptop routes.

Most tests need the laptop model models/p5_03_production (made by the base project's p5_03); without it they skip,
unless MLOPS_REQUIRE_MODEL=1 is set (see tests/conftest.py). The last 2 tests need no model.
Run from the ccfd-project folder: python -m pytest -q tests/test_serving_contract.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "models" / "p5_03_production"
PAYLOAD = PROJECT / "tests" / "payloads" / "sample_transactions.json"

ROW_FIELDS = {"prediction", "probability", "threshold", "risk_score", "model_name"}
JSON_LINES = {"Content-Type": "application/jsonlines"}
JSON = {"Content-Type": "application/json"}


@pytest.fixture(scope="module")
def laptop_model():
    if not (MODEL_DIR / "model_latest.pkl").exists():
        pytest.skip("no laptop model in models/p5_03_production: run the base project's chain up to p5_03_train_production_model.py")
    return MODEL_DIR


@pytest.fixture(scope="module")
def api():
    import m01_03_serving_api

    return m01_03_serving_api


@pytest.fixture(scope="module")
def client(api, laptop_model):
    # The tests always use the laptop model, whatever MODEL_DIR says in your shell
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "MODEL_DIR", laptop_model)
        with TestClient(api.app) as test_client:   # "with" runs the app's startup: the model is loaded once
            yield test_client


@pytest.fixture(scope="module")
def transactions():
    return json.loads(PAYLOAD.read_text())


@pytest.fixture(scope="module")
def expected_model_name(laptop_model):
    return json.loads((laptop_model / "metadata_latest.json").read_text())["model_name"]


def lines(*records):
    """A JSON Lines body: one JSON object per line."""
    return "".join(json.dumps(record) + "\n" for record in records)


# ---------------------------------------------------------------------------
# The SageMaker contract
# ---------------------------------------------------------------------------
def test_ping(client):
    assert client.get("/ping").status_code == 200


def test_one_json_object_gives_one_object(client, transactions):
    response = client.post("/invocations", json=transactions[1])
    assert response.status_code == 200
    row = response.json()
    assert isinstance(row, dict)
    assert set(row) == ROW_FIELDS


def test_json_list_gives_a_list(client, transactions):
    response = client.post("/invocations", json=transactions)
    assert response.status_code == 200
    rows = response.json()
    assert isinstance(rows, list) and len(rows) == 3
    assert all(set(row) == ROW_FIELDS for row in rows)


def test_json_lines_gives_one_line_per_transaction(client, transactions):
    response = client.post("/invocations", content=lines(*transactions), headers=JSON_LINES)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/jsonlines")
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert len(rows) == 3
    # the same answers as the JSON list
    as_list = client.post("/invocations", json=transactions).json()
    assert [r["probability"] for r in rows] == [r["probability"] for r in as_list]


def test_base_three_transactions_are_fraud(client, transactions):
    """The base project's p5_05 reference: 3/3 FRAUD at 60.60%, 99.84% and 99.60%."""
    rows = client.post("/invocations", json=transactions).json()
    assert [r["prediction"] for r in rows] == ["FRAUD", "FRAUD", "FRAUD"]
    assert [f"{r['probability']:.2%}" for r in rows] == ["60.60%", "99.84%", "99.60%"]


def test_model_name_and_threshold_in_every_row(client, transactions, expected_model_name):
    rows = client.post("/invocations", json=transactions).json()
    assert expected_model_name == "p5_03_production"
    for row in rows:
        assert row["model_name"] == expected_model_name
        assert row["threshold"] == 0.5
        assert row["risk_score"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# What the body may look like
# ---------------------------------------------------------------------------
def test_blank_lines_in_json_lines_are_skipped(client, transactions):
    body = json.dumps(transactions[0]) + "\n\n  \n" + json.dumps(transactions[1]) + "\n"
    response = client.post("/invocations", content=body, headers=JSON_LINES)
    assert response.status_code == 200
    assert len(response.text.splitlines()) == 2      # one result per non-blank line


@pytest.mark.parametrize(
    "content_type, lines_back",
    [
        ("application/json; charset=utf-8", False),
        ("APPLICATION/JSONLINES", True),
        ("application/x-jsonlines", True),
        ("application/jsonl", True),
        ("application/x-ndjson", True),
    ],
)
def test_content_type_variants(client, transactions, content_type, lines_back):
    body = lines(transactions[1]) if lines_back else json.dumps(transactions[1])
    response = client.post("/invocations", content=body, headers={"Content-Type": content_type})
    assert response.status_code == 200
    first = json.loads(response.text.splitlines()[0]) if lines_back else response.json()
    assert first["prediction"] == "FRAUD"


def test_no_content_type_counts_as_json(client, transactions):
    response = client.post("/invocations", content=json.dumps(transactions[1]).encode())
    assert response.status_code == 200
    assert response.json()["prediction"] == "FRAUD"


@pytest.mark.parametrize("content_type", ["text/csv", "text/plain", "application/x-www-form-urlencoded"])
def test_other_content_types_give_415(client, transactions, content_type):
    response = client.post("/invocations", content=json.dumps(transactions[1]),
                           headers={"Content-Type": content_type})
    assert response.status_code == 415
    assert "application/json or application/jsonlines" in response.json()["detail"]


@pytest.mark.parametrize("headers", [JSON, JSON_LINES], ids=["json", "jsonlines"])
def test_utf8_bom_is_accepted(client, transactions, headers):
    """Windows PowerShell 5.1 writes a byte order mark (BOM) at the start of a UTF-8 file."""
    body = b"\xef\xbb\xbf" + json.dumps(transactions[1]).encode()
    assert client.post("/invocations", content=body, headers=headers).status_code == 200


def test_line_separator_inside_a_string_stays_one_json_line(client, transactions):
    """JSON Lines records end with a newline only; U+2028 inside a string value is not a line end."""
    record = dict(transactions[1], note="a\u2028b")
    body = json.dumps(record, ensure_ascii=False) + "\r\n"     # a Windows line end too
    response = client.post("/invocations", content=body.encode("utf-8"), headers=JSON_LINES)
    assert response.status_code == 200
    assert len(response.text.splitlines()) == 1


# ---------------------------------------------------------------------------
# Bad input: HTTP 400 with the reason
# ---------------------------------------------------------------------------
def _changed(transactions, **changes):
    record = dict(transactions[1])
    record.update(changes)
    return record


@pytest.mark.parametrize(
    "make_body, reason",
    [
        (lambda t: {k: v for k, v in t[1].items() if k != "transaction_amount"}, "transaction_amount: Field required"),
        (lambda t: _changed(t, transaction_amount=-5), "transaction_amount"),
        (lambda t: _changed(t, transaction_hour=24), "transaction_hour"),
        (lambda t: _changed(t, merchant_category="casino"), "merchant_category"),
        (lambda t: [t[0], _changed(t, card_present="maybe")], "Transaction 1: card_present"),
        (lambda t: [], "no transactions"),
        (lambda t: "just a string", "Transaction 0"),
    ],
    ids=["missing-field", "negative-amount", "hour-24", "unknown-category", "bad-record-in-list", "empty-list",
         "not-an-object"],
)
def test_bad_json_gives_400_with_reason(client, transactions, make_body, reason):
    response = client.post("/invocations", json=make_body(transactions))
    assert response.status_code == 400
    assert reason in response.json()["detail"]


@pytest.mark.parametrize(
    "body, headers, reason",
    [
        ("this is not json", JSON, "The body is not valid JSON"),
        ("", JSON, "The body is not valid JSON"),
        ("", JSON_LINES, "no transactions"),
        ("\n  \n", JSON_LINES, "no transactions"),
        (b"\xff\xfe{}", JSON, "not UTF-8"),
    ],
    ids=["not-json", "empty-json", "empty-jsonlines", "blank-jsonlines", "not-utf8"],
)
def test_bodies_that_are_not_transactions_give_400(client, body, headers, reason):
    response = client.post("/invocations", content=body, headers=headers)
    assert response.status_code == 400
    assert reason in response.json()["detail"]


def test_bad_json_line_gives_400_with_line_number(client, transactions):
    body = json.dumps(transactions[0]) + "\n{not json\n"
    response = client.post("/invocations", content=body, headers=JSON_LINES)
    assert response.status_code == 400
    assert "Line 2 is not valid JSON" in response.json()["detail"]


def test_bad_record_in_json_lines_names_its_line(client, transactions):
    """Blank lines count, so the number points at the line in the file (Batch Transform input)."""
    body = json.dumps(transactions[0]) + "\n\n" + json.dumps(_changed(transactions, transaction_hour=99)) + "\n"
    response = client.post("/invocations", content=body, headers=JSON_LINES)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Line 3: transaction_hour")


def _with_number(transactions, field, number_text):
    """A JSON text of transaction 1 with one field written as raw JSON number text."""
    text = json.dumps(_changed(transactions, **{field: 1}))
    return text.replace(f'"{field}": 1', f'"{field}": {number_text}')


NOT_FINITE = [
    ("transaction_amount", "1e309", "transaction_amount: must be a finite number"),
    ("avg_transaction_amount", "Infinity", "avg_transaction_amount: must be a finite number"),
    ("transaction_amount", "NaN", "transaction_amount: Input should be greater than 0"),
    ("days_since_last_txn", "9" * 400, "days_since_last_txn: must be a finite number"),
    ("transaction_count_24h", "9" * 400, "transaction_count_24h: must be a finite number"),
]
NOT_FINITE_IDS = ["1e309", "infinity", "nan", "400-digit-days", "400-digit-count"]


@pytest.mark.parametrize("field, number_text, reason", NOT_FINITE, ids=NOT_FINITE_IDS)
def test_numbers_the_model_cannot_use_give_400_in_a_list(client, transactions, field, number_text, reason):
    body = "[" + json.dumps(transactions[0]) + ", " + _with_number(transactions, field, number_text) + "]"
    response = client.post("/invocations", content=body, headers=JSON)
    assert response.status_code == 400
    assert response.json()["detail"] == f"Transaction 1: {reason}"


@pytest.mark.parametrize("field, number_text, reason", NOT_FINITE, ids=NOT_FINITE_IDS)
def test_numbers_the_model_cannot_use_give_400_in_json_lines(client, transactions, field, number_text, reason):
    body = json.dumps(transactions[0]) + "\n" + _with_number(transactions, field, number_text) + "\n"
    response = client.post("/invocations", content=body, headers=JSON_LINES)
    assert response.status_code == 400
    assert response.json()["detail"] == f"Line 2: {reason}"


@pytest.mark.parametrize("headers, where", [(JSON, "The body"), (JSON_LINES, "Line 1")], ids=["json", "jsonlines"])
def test_integer_over_4300_digits_gives_400(client, transactions, headers, where):
    body = _with_number(transactions, "days_since_last_txn", "9" * 5000)
    response = client.post("/invocations", content=body, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"].startswith(f"{where} is not valid JSON: Exceeds the limit (4300 digits)")


def test_infinite_amount_on_inference_gives_400(client, transactions):
    response = client.post("/inference", content=_with_number(transactions, "transaction_amount", "1e309"),
                           headers=JSON)
    assert response.status_code == 400
    assert response.json()["detail"] == "Transaction 0: transaction_amount: must be a finite number"


def test_bad_input_on_inference_gives_400_not_422(client, transactions):
    response = client.post("/inference", json=_changed(transactions, international="maybe"))
    assert response.status_code == 400
    assert "international" in response.json()["detail"]


# ---------------------------------------------------------------------------
# The laptop routes
# ---------------------------------------------------------------------------
def test_inference_keeps_base_answer_in_ascii(client, transactions):
    response = client.post("/inference", json=transactions[1])
    assert response.status_code == 200
    answer = response.json()
    assert set(answer) == ROW_FIELDS | {"confidence", "message"}
    assert answer["message"].startswith("HIGH RISK: 99.8% probability of fraud")
    assert answer["message"].isascii()


def test_health(client, expected_model_name):
    answer = client.get("/health").json()
    assert answer["status"] == "healthy"
    assert answer["model_loaded"] is True
    assert answer["model_name"] == expected_model_name
    assert answer["model_type"] == "LogisticRegression"


def test_ui_names_the_tuned_model(client, expected_model_name):
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert f"Tuned Model (best config) <strong>{expected_model_name}</strong>" in page
    assert "Baseline Model" not in page
    assert "__MODEL_NAME__" not in page
    assert page.isascii()


# ---------------------------------------------------------------------------
# Where the model comes from, and no MLflow
# ---------------------------------------------------------------------------
def test_loads_from_the_sagemaker_layout(api, laptop_model, tmp_path):
    """SageMaker unpacks model.tar.gz into /opt/ml/model; the same layout in a temporary folder."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("model_latest.pkl", "metadata_latest.json"):
        shutil.copy(laptop_model / name, model_dir / name)
    pipeline, metadata = api.load_model(model_dir)
    assert hasattr(pipeline, "predict_proba")
    assert metadata["model_name"] == "p5_03_production"


# The last 2 tests need no model: they also run in a fresh clone
def test_empty_model_folder_fails_clearly(api, tmp_path):
    with pytest.raises(FileNotFoundError):
        api.load_model(tmp_path / "model")


def test_no_mlflow_in_the_serving_path():
    """The serving image has no MLflow: importing the API must not import it."""
    code = "import sys, m01_03_serving_api; print('mlflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
