"""
The Net Benefit gate (one business metric decides every promotion).

The rule, in one line: score the champion and the candidate on the SAME fixed data with the base project's
calculate_business_metrics, and PROMOTE the candidate only if its Net Benefit is strictly higher.
A tie keeps the champion. With no champion yet, the first candidate is promoted.

The decision is a dict (and, if you ask, a JSON file) in the shape of the original monitoring study
(the original promotion_decision.json): both models' numbers, the gain, the decision and
the model to serve. The base project called the serving model "current" and the outcomes PROMOTE / KEEP; this course
calls it "champion" (the MLflow alias) and the outcomes PROMOTE / REJECT. This course adds one field,
"promote": 1 or 0. The SageMaker pipeline's condition step reads that field.
What this course leaves out of the base project's record: holdout_batch/_rows/_frauds are renamed gate_data/_rows/_frauds; the per-model
versions_on_disk, created_at, trained_on and at_0.9 and the top-level net_benefit_gain_at_0.9 are dropped. The base project scored
each model at its own saved threshold (0.5 for both there); this course uses the fixed 0.5 for both (MLOps 05 Step-02).

    from mlops_utils import gate
    data = gate.load_gate_data("data/gate_holdout.csv")     # or gate.base_test_set(...)
    record = gate.run_gate(candidate_model, champion_model, data,
                           candidate_name="ccfd-fraud-model", candidate_version="2",
                           champion_name="ccfd-fraud-model", champion_version="1",
                           data_name="gate_holdout.csv", out_path="gate_decision.json")
    print(record["decision"], record["promote"])

This file does not import MLflow, so it also runs inside the training image (MLOps 13).
"""

import contextlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# The base project's code, unchanged
from utils.data_utils import split_data
from utils.feature_engineering_utils import apply_feature_engineering
from utils.metrics_utils import calculate_business_metrics

THRESHOLD = 0.5          # the base project's DEFAULT_THRESHOLD in p5_04: FRAUD when the probability is 0.5 or more
TARGET = "is_fraud"      # the base project's label column
BASE_FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"   # the base project's FE1 data (p4_01 output)


# ---------------------------------------------------------------------------
# 1. The fixed data both models are scored on
# ---------------------------------------------------------------------------
def add_amount_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """Add the base project's FE1 column amount_ratio when it is missing (raw files like the gate holdout do not have it)."""
    if "amount_ratio" in data.columns:
        return data
    with contextlib.redirect_stdout(io.StringIO()):   # the base project's helper prints a progress line; keep our output short
        return apply_feature_engineering(data)


def load_gate_data(csv_path: str | Path) -> pd.DataFrame:
    """Read a fixed gate file (for example data/gate_holdout.csv or month_4) and add amount_ratio if needed."""
    return add_amount_ratio(pd.read_csv(csv_path))


def base_test_set(fe_csv: str | Path = BASE_FE_FILE) -> pd.DataFrame:
    """
    The base project's own test set: the FE1 file split with the base project's split_data (test_size 0.2, seed 42, stratified).
    On data v1 that is 2,000 rows with 60 frauds; it gives the base project's Improvement Journey numbers
    (p3_01 39,625, p5_01 37,025, p5_03 50,550).
    """
    with contextlib.redirect_stdout(io.StringIO()):   # the base project's split_data prints emoji progress lines
        _, X_test, _, y_test, _ = split_data(pd.read_csv(fe_csv))
    return X_test.assign(**{TARGET: y_test})


# ---------------------------------------------------------------------------
# 2. Score one model with the base project's numbers
# ---------------------------------------------------------------------------
def score(model, data: pd.DataFrame, threshold: float = THRESHOLD) -> dict:
    """
    The base project's metrics for one model on one data set: the business numbers from calculate_business_metrics
    plus F1, precision, recall and ROC AUC (as the base project's print_metrics shows them).
    The model is any object with predict_proba, for example the base project's sklearn Pipeline.
    """
    y_true = data[TARGET].to_numpy()
    probability = model.predict_proba(data.drop(columns=[TARGET]))[:, 1]
    y_pred = (probability >= threshold).astype(int)          # the same rule the serving API uses
    business = calculate_business_metrics(y_true, y_pred)
    return {
        "threshold": threshold,
        "net_benefit": int(business["net_benefit"]),
        "fraud_caught": int(business["fraud_caught"]),
        "fraud_missed": int(business["fraud_missed"]),
        "false_alarms": int(business["false_alarms"]),
        "true_negatives": int(business["true_negatives"]),
        "total_saved": int(business["total_saved"]),
        "total_cost": int(business["total_cost"]),
        "flagged_share": round(float(y_pred.mean()), 4),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(set(y_true)) == 2 else None,
    }


# ---------------------------------------------------------------------------
# 3. The rule
# ---------------------------------------------------------------------------
def decide(candidate_net_benefit: int, champion_net_benefit: int | None) -> tuple[str, int]:
    """PROMOTE (1) only if the candidate is strictly better; a tie or a lower number is REJECT (0)."""
    if champion_net_benefit is None or candidate_net_benefit > champion_net_benefit:
        return "PROMOTE", 1
    return "REJECT", 0


def run_gate(candidate, champion, data: pd.DataFrame, *, candidate_name: str, candidate_version: str,
             champion_name: str | None = None, champion_version: str | None = None,
             data_name: str = "gate data", out_path: str | Path | None = None) -> dict:
    """
    Score both models on the same data, decide, and return the decision record.
    champion=None means there is no champion yet: the candidate is promoted.
    out_path: also write the record as JSON there (for example gate_decision.json).
    """
    cand = {"model_name": candidate_name, "model_version": str(candidate_version), **_short(score(candidate, data))}
    champ = None
    if champion is not None:
        champ = {"model_name": champion_name, "model_version": str(champion_version), **_short(score(champion, data))}

    decision, promote = decide(cand["net_benefit"], champ["net_benefit"] if champ else None)
    gain = cand["net_benefit"] - champ["net_benefit"] if champ else None
    serve = cand if promote else champ

    if champ is None:
        notes = ["No champion yet, so the first candidate is promoted."]
        reason = f"no champion yet; candidate {cand['net_benefit']:,}"
    elif gain == 0:
        notes = ["A tie keeps the champion: the candidate must be strictly better."]
        reason = f"candidate {cand['net_benefit']:,} = champion {champ['net_benefit']:,} (a tie keeps the champion)"
    elif promote:
        notes = []
        reason = f"candidate {cand['net_benefit']:,} > champion {champ['net_benefit']:,} (+{gain:,})"
    else:
        notes = ["The candidate did not beat the champion, so nothing changes."]
        reason = f"candidate {cand['net_benefit']:,} <= champion {champ['net_benefit']:,} ({gain:,})"

    record = {
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),    # UTC (+00:00): laptop and job agree
        "gate_data": data_name,
        "gate_rows": len(data),
        "gate_frauds": int(data[TARGET].sum()),
        "rule": f"promote only if candidate net_benefit > champion net_benefit on {data_name} "
                f"(strictly better; a tie keeps the champion), threshold {THRESHOLD} for both",
        "champion": champ,
        "candidate": cand,
        "net_benefit_gain": gain,
        "decision": decision,
        "promote": promote,
        "reason": reason,
        "serve_model_name": serve["model_name"],
        "serve_model_version": serve["model_version"],
        "notes": notes,
    }
    if out_path is not None:
        write_decision(record, out_path)
    return record


def write_decision(record: dict, path: str | Path) -> Path:
    """Write the decision record as readable JSON (the pipeline reads the 'promote' field)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def version_tags(record: dict) -> dict[str, str]:
    """The decision as MLflow model-version tags (MLOps 05 writes them on the candidate version)."""
    champion = record["champion"]
    return {
        "gate_result": record["decision"],
        "gate_reason": record["reason"],
        "candidate_net_benefit": str(record["candidate"]["net_benefit"]),
        "champion_net_benefit": str(champion["net_benefit"]) if champion else "none",
        "net_benefit_gain": str(record["net_benefit_gain"]) if champion else "none",
        "decided_at": record["decided_at"],
    }


def _short(metrics: dict) -> dict:
    """The per-model numbers the decision record keeps: a subset of the original monitoring study per-model keys (the 0.9 block
    and the disk fields versions_on_disk, created_at, trained_on are left out)."""
    keys = ("threshold", "net_benefit", "fraud_caught", "fraud_missed", "false_alarms", "flagged_share")
    return {key: metrics[key] for key in keys}
