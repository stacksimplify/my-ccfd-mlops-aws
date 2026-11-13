"""
MLOps 01: the serving API for the fraud model.

One small FastAPI app that serves the tuned model of the base project (p5_03_production) in two ways:

1. The SageMaker contract. The same container runs on a real-time endpoint, a serverless endpoint
   and a Batch Transform job; SageMaker calls only these 2 routes on port 8080:
     GET  /ping          200 when the model is loaded (SageMaker's health check)
     POST /invocations   Content-Type application/json: one transaction (answer: one object)
                         or a list of transactions (answer: a list)
                         Content-Type application/jsonlines: one transaction per line
                         (answer: one result per line; Batch Transform uses this)
                         No Content-Type counts as application/json; any other type gets HTTP 415
2. The laptop routes we know from the base project's p5_04:
     POST /inference     one transaction (the /ui page calls it)
     GET  /health        is the model loaded, and which one
     GET  /ui            the web page (templates/index_course.html)

What we reuse from the base project's p5_04_inference_api.py (imported, never changed):
  TransactionRequest     the 8 input fields and their checks
  prepare_input          turns one transaction into a DataFrame and adds amount_ratio (FE1)
  calculate_risk_score   LOW, MEDIUM, HIGH or CRITICAL
  DEFAULT_THRESHOLD      0.5

Each result row has 5 fields: prediction, probability, threshold, risk_score, model_name.
model_name comes from metadata_latest.json next to the model.

The model comes only from the folder in the environment variable MODEL_DIR:
  laptop default   models/p5_03_production (made by the base project's p5_03_train_production_model.py)
  on SageMaker     /opt/ml/model (SageMaker unpacks model.tar.gz there)
There is no MLflow here, and the app writes no file: a serverless endpoint has a read-only file system
(only /tmp is writable).

Bad input gets HTTP 400 with the reason, on every route. The reason names the transaction:
"Transaction 2: ..." in a JSON list, "Line 3: ..." (the line in the body) in JSON Lines.
On top of the base project's checks, this API refuses numbers the model cannot use (Infinity, 1e309, an integer with
hundreds of digits); the base project's TransactionRequest stays unchanged.

Run it from the ccfd-project folder (port 8080, this machine only):
    export MODEL_DIR=models/p5_03_production
    python m01_03_serving_api.py
or
    python -m uvicorn m01_03_serving_api:app --port 8080
In the container the serve launcher starts it on 0.0.0.0:8080.
"""

import contextlib
import html
import io
import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

# The base project's code, unchanged
from p5_04_inference_api import (
    DEFAULT_THRESHOLD,
    TransactionRequest,
    calculate_risk_score,
    prepare_input,
)
from utils.model_utils import load_pipeline

# ============================================================================
# Settings
# ============================================================================
HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "models" / "p5_03_production"))
UI_TEMPLATE = HERE / "templates" / "index_course.html"

# The content types POST /invocations accepts (anything else gets HTTP 415)
JSON_TYPES = {"application/json"}
JSON_LINES_TYPES = {"application/jsonlines", "application/x-jsonlines", "application/jsonl", "application/x-ndjson"}

# The 5 number fields of the base project's TransactionRequest
NUMBER_FIELDS = ("transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
                 "transaction_count_24h")

# Filled once when the server starts (see lifespan below)
PIPELINE = None
METADATA = {}


# ============================================================================
# Load the model
# ============================================================================
def load_model(model_dir: Path):
    """Load model_latest.pkl and metadata_latest.json from one folder with the base project's load_pipeline."""
    # The base project's load_pipeline(model_name, save_dir) reads <save_dir>/<model_name>/model_latest.pkl,
    # so we pass the last part of the folder and its parent:
    # /opt/ml/model -> load_pipeline("model", save_dir="/opt/ml")
    # The base project's loader prints emoji; we hide that output so our logs stay plain ASCII.
    with contextlib.redirect_stdout(io.StringIO()):
        pipeline, metadata = load_pipeline(model_dir.name, save_dir=str(model_dir.parent))
    return pipeline, metadata


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, when the server starts. No model means the server does not start."""
    global PIPELINE, METADATA
    try:
        PIPELINE, METADATA = load_model(MODEL_DIR)
    except FileNotFoundError:
        print(f"ERROR: no model_latest.pkl in {MODEL_DIR}. Set MODEL_DIR to the model folder.", flush=True)
        raise
    print(f"Model loaded: {model_name()} ({METADATA.get('model_type', 'unknown')}) from {MODEL_DIR}", flush=True)
    yield
    print("Shutting down the serving API.", flush=True)


app = FastAPI(
    title="Credit Card Fraud Detection: serving API",
    description="The base project's tuned model behind the SageMaker contract: GET /ping and POST /invocations",
    lifespan=lifespan,
)


# ============================================================================
# Helpers
# ============================================================================
def model_name() -> str:
    """The model's name from metadata_latest.json (MLOps 12's canary watcher reads it per row)."""
    return METADATA.get("model_name", "unknown")


def bad_input(reason: str) -> HTTPException:
    """HTTP 400 with the reason, for anything wrong in the request."""
    return HTTPException(status_code=400, detail=reason)


def error_reason(errors: list) -> str:
    """Turn pydantic's error list into one readable line, for example
    'transaction_amount: Input should be greater than 0'."""
    parts = []
    for error in errors:
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        parts.append(f"{field}: {error['msg']}" if field else error["msg"])
    return "; ".join(parts)


def parse_json(text: str, problem: str):
    """json.loads, but anything that is not valid JSON gets HTTP 400 instead of a crash."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise bad_input(f"{problem}: {err.msg}")
    except (ValueError, RecursionError) as err:   # an integer longer than 4,300 digits, or very deep nesting
        raise bad_input(f"{problem}: {str(err).split(':')[0]}")


def read_records(body: bytes, json_lines: bool) -> tuple[list, bool]:
    """Turn the request body into a list of (label, raw transaction) pairs.

    The label names the transaction in error messages: "Transaction 0", "Transaction 1", ... for JSON,
    "Line 3" (the line number in the body, blank lines counted) for JSON Lines.
    Returns (items, answer_with_list). One JSON object gets one object back;
    a JSON list or JSON Lines get one result per transaction.
    """
    try:
        text = body.decode("utf-8-sig")   # utf-8-sig also accepts the BOM that some Windows tools write first
    except UnicodeDecodeError:
        raise bad_input("The body is not UTF-8 text.")

    if json_lines:
        items = []
        # JSON Lines records end with "\n" only (str.splitlines would also cut a string that holds U+2028)
        for number, line in enumerate(text.split("\n"), start=1):
            line = line.removesuffix("\r")   # Windows line ends
            if line.strip():
                items.append((f"Line {number}", parse_json(line, f"Line {number} is not valid JSON")))
        return items, True

    data = parse_json(text, "The body is not valid JSON")
    if isinstance(data, list):
        return [(f"Transaction {index}", record) for index, record in enumerate(data)], True
    return [("Transaction 0", data)], False


def check_numbers(transaction: TransactionRequest, label: str) -> None:
    """The base project's checks let some numbers through that the model cannot use: Infinity, 1e309 (too big for a float,
    so it becomes Infinity) and integers with hundreds of digits. This API refuses them here with HTTP 400."""
    for field in NUMBER_FIELDS:
        try:
            finite = math.isfinite(float(getattr(transaction, field)))
        except OverflowError:   # an integer too big for a float
            finite = False
        if not finite:
            raise bad_input(f"{label}: {field}: must be a finite number")


def validate(items: list) -> list[TransactionRequest]:
    """Check every transaction with the base project's TransactionRequest (the 8 fields and their limits), then its numbers."""
    if not items:
        raise bad_input("The request has no transactions.")
    transactions = []
    for label, record in items:
        try:
            transaction = TransactionRequest.model_validate(record)
        except ValidationError as err:
            raise bad_input(f"{label}: {error_reason(err.errors())}")
        check_numbers(transaction, label)
        transactions.append(transaction)
    return transactions


def predict(transactions: list[TransactionRequest]) -> list[dict]:
    """Score the transactions in one call and return one result row per transaction."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="The model is not loaded.")
    # The base project's prepare_input makes one row per transaction and adds amount_ratio
    rows = pd.concat([prepare_input(t) for t in transactions], ignore_index=True)
    probabilities = PIPELINE.predict_proba(rows)[:, 1]   # column 1 = probability of fraud

    results = []
    for probability in probabilities:
        probability = float(probability)
        results.append({
            "prediction": "FRAUD" if probability >= DEFAULT_THRESHOLD else "LEGITIMATE",
            "probability": probability,
            "threshold": DEFAULT_THRESHOLD,
            "risk_score": calculate_risk_score(probability),
            "model_name": model_name(),
        })
    return results


# FastAPI answers a bad /inference body with 422 by default; we answer 400 with the reason, like /invocations
@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, err: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": error_reason(err.errors())})


# ============================================================================
# The SageMaker contract: GET /ping and POST /invocations
# ============================================================================
@app.get("/ping")
async def ping():
    """SageMaker's health check: 200 when the model is loaded."""
    return Response(status_code=200 if PIPELINE is not None else 503)


@app.post("/invocations")
async def invocations(request: Request):
    """Score one transaction, a list, or JSON Lines (see the top of this file)."""
    # "application/json; charset=utf-8" counts as application/json; no Content-Type header counts too
    content_type = request.headers.get("content-type", "application/json").split(";")[0].strip().lower()
    if content_type not in JSON_TYPES | JSON_LINES_TYPES:
        raise HTTPException(status_code=415, detail=f"Content-Type {content_type} is not supported: "
                                                    "send application/json or application/jsonlines.")
    json_lines = content_type in JSON_LINES_TYPES

    items, answer_with_list = read_records(await request.body(), json_lines)
    results = predict(validate(items))

    if json_lines:
        text = "".join(json.dumps(row) + "\n" for row in results)
        return Response(content=text, media_type="application/jsonlines")
    return JSONResponse(content=results if answer_with_list else results[0])


# ============================================================================
# The laptop routes: /inference, /health, /ui
# ============================================================================
@app.post("/inference")
async def inference(transaction: TransactionRequest):
    """One transaction in, one answer out, with the base project's confidence and a message (the /ui page shows both)."""
    check_numbers(transaction, "Transaction 0")
    result = predict([transaction])[0]
    probability = result["probability"]
    result["confidence"] = min(abs(probability - DEFAULT_THRESHOLD) * 2, 1.0)
    if result["prediction"] == "FRAUD":
        result["message"] = f"HIGH RISK: {probability:.1%} probability of fraud. Recommend blocking."
    else:
        result["message"] = f"LOW RISK: {1 - probability:.1%} confidence legitimate. Safe to approve."
    return result


@app.get("/health")
async def health():
    """Is the model loaded, and which one?"""
    return {
        "status": "healthy" if PIPELINE is not None else "unhealthy",
        "model_loaded": PIPELINE is not None,
        "model_name": model_name(),
        "model_type": METADATA.get("model_type", "unknown"),
        "model_dir": str(MODEL_DIR),
    }


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """The web page. Read as UTF-8 on every operating system; it names the model it serves."""
    try:
        page = UI_TEMPLATE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: templates/index_course.html not found</h1>", status_code=500)
    return HTMLResponse(content=page.replace("__MODEL_NAME__", html.escape(model_name())))


# ============================================================================
# Run it directly: python m01_03_serving_api.py (port 8080, like the base project's "python p5_04_inference_api.py" on 8001)
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1: only this machine can call it (your laptop reaches it through the SSH port forward;
    # the container's serve launcher listens on 0.0.0.0)
    uvicorn.run(app, host="127.0.0.1", port=8080)
