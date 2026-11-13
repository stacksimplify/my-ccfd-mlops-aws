"""
MLOps 15: our own monitoring (no SageMaker Model Monitor: closed to new customers).

1. Drift: the original monitoring study's PSI (monitoring_utils.py, ported unchanged so the numbers match the base project's findings): 10 quantile bins
   of the reference for a numeric feature, category shares for a categorical one; bands OK < 0.10 <= MODERATE < 0.25 <=
   SIGNIFICANT.
2. Capture: SageMaker data capture files (JSON Lines). Each record holds endpointInput and endpointOutput, each with
   data that is JSON text or BASE64 of JSON; one record may hold a list of transactions.
3. Late labels: join captured predictions with the true labels by InferenceId "<tag>:month_<k>:<row_id>".
4. Net Benefit by month with the base project's calculate_business_metrics.
"""

import base64
import json

import numpy as np
import pandas as pd

from utils.metrics_utils import calculate_business_metrics

_EPSILON = 1e-6                     # keeps log() finite when a bin is empty
FIELDS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
          "merchant_category", "card_present", "international", "transaction_count_24h"]


# ---------------------------------------------------------------------------
# 1. PSI (the original monitoring study utils/monitoring_utils.py, unchanged logic)
# ---------------------------------------------------------------------------
def population_stability_index(reference, current, buckets=10):
    """PSI for one numeric feature: quantile bins of the reference, open outer edges."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_share = np.clip(ref_counts / max(len(reference), 1), _EPSILON, None)
    cur_share = np.clip(cur_counts / max(len(current), 1), _EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_categorical(reference, current):
    """PSI for one categorical feature (the share of each category)."""
    ref, cur = pd.Series(reference).astype(str), pd.Series(current).astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_share = np.clip(ref.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(), _EPSILON, None)
    cur_share = np.clip(cur.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(), _EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_band(psi):
    """OK < 0.10 <= MODERATE < 0.25 <= SIGNIFICANT."""
    if psi < 0.10:
        return "OK"
    if psi < 0.25:
        return "MODERATE"
    return "SIGNIFICANT"


def drift_report(reference_df, current_df, categorical_cols, numerical_cols):
    """One row per feature (feature, type, psi, band), worst first."""
    rows = []
    for col in numerical_cols:
        if col in reference_df and col in current_df:
            psi = population_stability_index(reference_df[col], current_df[col])
            rows.append({"feature": col, "type": "numeric", "psi": round(psi, 4), "band": psi_band(psi)})
    for col in categorical_cols:
        if col in reference_df and col in current_df:
            psi = psi_categorical(reference_df[col], current_df[col])
            rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4), "band": psi_band(psi)})
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# 2. Data capture records
# ---------------------------------------------------------------------------
def capture_payload(part: dict):
    """The JSON of endpointInput or endpointOutput, whether its encoding is JSON or BASE64."""
    data = part["data"]
    if part.get("encoding") == "BASE64":
        data = base64.b64decode(data).decode()
    text = data.strip()
    if "\n" in text and not text.startswith("["):          # JSON Lines: one object per line
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def capture_rows(records) -> pd.DataFrame:
    """One row per captured transaction: inference_id, inference_time, the 8 fields, prediction, probability."""
    rows = []
    for record in records:
        meta = record.get("eventMetadata", {})
        inputs = capture_payload(record["captureData"]["endpointInput"])
        outputs = capture_payload(record["captureData"]["endpointOutput"])
        inputs = inputs if isinstance(inputs, list) else [inputs]
        outputs = outputs if isinstance(outputs, list) else [outputs]
        for transaction, answer in zip(inputs, outputs, strict=False):
            rows.append({"inference_id": meta.get("inferenceId", ""), "inference_time": meta.get("inferenceTime", ""),
                         **{f: transaction.get(f) for f in FIELDS},
                         "prediction": answer.get("prediction"), "probability": answer.get("probability")})
    return pd.DataFrame(rows)


def read_capture_lines(texts) -> pd.DataFrame:
    """capture_rows() over the text of one or more capture .jsonl files."""
    return capture_rows(json.loads(line) for text in texts for line in text.splitlines() if line.strip())


def month_of(inference_id: str) -> str:
    """'<tag>:month_3:000042' -> 'month_3' ('' when the id does not follow that pattern)."""
    parts = str(inference_id).split(":")
    return parts[1] if len(parts) == 3 and parts[1].startswith("month_") else ""


# ---------------------------------------------------------------------------
# 3. + 4. Late labels and Net Benefit by month
# ---------------------------------------------------------------------------
def join_ground_truth(captured: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Captured predictions joined with the late labels (columns inference_id, is_fraud) by InferenceId."""
    joined = captured.merge(labels[["inference_id", "is_fraud"]], on="inference_id", how="inner")
    joined["month"] = joined["inference_id"].map(month_of)
    return joined


def net_benefit_by_month(joined: pd.DataFrame) -> dict[str, int]:
    """{'month_1': 76450, ...}: the base project's Net Benefit of the captured predictions, per month."""
    result = {}
    for month, group in sorted(joined.groupby("month")):
        if month:
            y_pred = (group["prediction"] == "FRAUD").astype(int).tolist()
            result[month] = int(calculate_business_metrics(group["is_fraud"].astype(int).tolist(), y_pred)["net_benefit"])
    return result


def drop_pct(baseline: float, value: float) -> float:
    """How far Net Benefit fell from the baseline month, in percent (0 when it did not fall)."""
    return round(max(0.0, (baseline - value) / baseline * 100), 1) if baseline > 0 else 0.0
