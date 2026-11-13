"""
MLOps 12: the shared steps of the m12 scripts (endpoint ccfd-prod).

  probe_set()           the labeled probe set: the gate holdout of MLOps 09 (data/gate_holdout.csv, 2,000 rows,
                        seed 301, 60 frauds). No model of the course trained on these rows, and MLOps 09 decided on
                        them ($36,975 for the champion, $19,925 for the data v2 retrain), so the release numbers are
                        fair and the same as the gate's.
  score(variant)        one probe call: the 2,000 rows (no labels) as one JSON Lines request; returns
                        (InvokedProductionVariant, model_name, Net Benefit) scored with the base project's money rule
  variant(...)          one production variant (ml.m5.large)
  make_config(...)      create an endpoint config once (configs never change)
  already_serves(name)  True when ccfd-prod serves that config already (a second run changes nothing)
  say(text)             print with the time, through the account ID mask (the m12 result lines)
  log(text)             print through the mask, no time (for the mlops_utils.endpoints helpers: their wait lines
                        carry their own time)
"""

import json
import time
from functools import cache
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from mlops_utils import config
from utils.metrics_utils import calculate_business_metrics

EP = "ccfd-prod"
FIELDS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
          "merchant_category", "card_present", "international", "transaction_count_24h"]
PROBE = Path("data/gate_holdout.csv")
PROBE_COMMAND = ("python m03_03_simulate_new_data.py --mode refresh --rows 2000 --seed 301 "
                 "--output data/gate_holdout.csv")
sm = boto3.client("sagemaker", region_name=config.REGION)
runtime = boto3.client("sagemaker-runtime", region_name=config.REGION)


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S"), config.mask(str(text)), flush=True)


def log(text: str) -> None:
    print(config.mask(str(text)), flush=True)


@cache
def probe_set():
    if not PROBE.is_file():
        raise SystemExit(f"STOP: no probe set here ({PROBE}). Make it from the ccfd-project folder: {PROBE_COMMAND}")
    data = pd.read_csv(PROBE)
    body = "\n".join(json.dumps(row) for row in data[FIELDS].to_dict("records")) + "\n"
    return body, [int(label) for label in data["is_fraud"]]


def score(variant: str | None = None):
    body, labels = probe_set()
    request = {"EndpointName": EP, "ContentType": "application/jsonlines", "Accept": "application/jsonlines",
               "Body": body}
    if variant:
        request["TargetVariant"] = variant
    answer = runtime.invoke_endpoint(**request)
    rows = [json.loads(line) for line in answer["Body"].read().decode().splitlines() if line.strip()]
    predicted = [1 if row["prediction"] == "FRAUD" else 0 for row in rows]
    net_benefit = int(calculate_business_metrics(labels, predicted)["net_benefit"])
    return answer.get("InvokedProductionVariant"), rows[0].get("model_name"), net_benefit


def variant(name: str, model: str, count: int = 2, weight: float = 1.0) -> dict:
    return {"VariantName": name, "ModelName": model, "InstanceType": "ml.m5.large", "InitialInstanceCount": count,
            "InitialVariantWeight": weight}


def make_config(name: str, variants: list, shadow: list | None = None) -> str:
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        say(f"config {name}: exists (kept)")
        return name
    except ClientError:
        pass
    request = {"EndpointConfigName": name, "Tags": config.load().aws_tags(), "ProductionVariants": variants}
    if shadow:
        request["ShadowProductionVariants"] = shadow
    sm.create_endpoint_config(**request)
    say(f"config {name}: " + ", ".join(f"{v['VariantName']}={v['ModelName']} x{v['InitialInstanceCount']}"
                                         for v in variants + (shadow or [])))
    return name


def model_exists(name: str) -> bool:
    try:
        sm.describe_model(ModelName=name)
        return True
    except ClientError:
        return False


def current_config() -> str:
    return sm.describe_endpoint(EndpointName=EP)["EndpointConfigName"]


def already_serves(name: str) -> bool:
    """True (and one line) when ccfd-prod serves this config already: SageMaker refuses an update to it."""
    if current_config() != name:
        return False
    status = sm.describe_endpoint(EndpointName=EP)["EndpointStatus"]
    say(f"{EP} already serves {name} ({status}): nothing to update")
    return True
