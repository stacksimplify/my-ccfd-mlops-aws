"""MLOps 15: mlops_utils/monitoring.py, the drift job and the replay ids. No AWS."""
import base64
import json

import pandas as pd
import pytest

import m15_03_drift_job as drift
from mlops_utils import monitoring as m


def test_psi_bands_and_identical_data():
    assert m.psi_band(0.05) == "OK" and m.psi_band(0.1957) == "MODERATE" and m.psi_band(0.3) == "SIGNIFICANT"
    assert m.population_stability_index(range(1000), range(1000)) == pytest.approx(0, abs=1e-9)
    assert m.psi_categorical(["a", "b"] * 50, ["a", "b"] * 50) == pytest.approx(0, abs=1e-9)


def test_shifted_data_drifts():
    assert m.population_stability_index(range(1000), [x + 300 for x in range(1000)]) > 0.25


def record(inference_id, encoding, prediction="FRAUD"):
    transaction = json.dumps([{f: 1 for f in m.FIELDS}])
    answer = json.dumps([{"prediction": prediction, "probability": 0.9}])
    if encoding == "BASE64":
        transaction, answer = (base64.b64encode(t.encode()).decode() for t in (transaction, answer))
    return {"captureData": {"endpointInput": {"data": transaction, "encoding": encoding},
                            "endpointOutput": {"data": answer, "encoding": encoding}},
            "eventMetadata": {"inferenceId": inference_id, "inferenceTime": "t"}}


def test_capture_reader_reads_json_and_base64_records():
    rows = m.capture_rows([record("r:month_1:000001", "JSON"), record("r:month_3:000002", "BASE64", "LEGITIMATE")])
    assert list(rows["prediction"]) == ["FRAUD", "LEGITIMATE"]
    assert list(rows["inference_id"].map(m.month_of)) == ["month_1", "month_3"]


def test_labels_join_and_net_benefit_by_month():
    captured = m.capture_rows([record("r:month_1:000001", "JSON"), record("r:month_1:000002", "JSON", "LEGITIMATE")])
    labels = pd.DataFrame({"inference_id": ["r:month_1:000001", "r:month_1:000002"], "is_fraud": [1, 0]})
    joined = m.join_ground_truth(captured, labels)
    assert len(joined) == 2 and set(m.net_benefit_by_month(joined)) == {"month_1"}


def test_drop_pct():
    assert m.drop_pct(76450, 36525) == 52.2 and m.drop_pct(76450, 80000) == 0.0 and m.drop_pct(0, 5) == 0.0


def test_publish_sends_one_call_per_month_without_dimensions():
    sent = []

    class FakeCloudWatch:
        def put_metric_data(self, Namespace, MetricData):
            sent.append((Namespace, MetricData))

    summary = {"months": {"month_1": {"max_psi": 0.0061, "net_benefit": 76450, "net_benefit_drop_pct": 0.0},
                          "month_3": {"max_psi": 0.1957, "net_benefit": 36525, "net_benefit_drop_pct": 52.2}}}
    drift.publish(summary, FakeCloudWatch())
    assert [ns for ns, _ in sent] == ["MLOps/CCFD", "MLOps/CCFD"]
    assert all("Dimensions" not in d for _, data in sent for d in data)
    assert {d["MetricName"] for d in sent[1][1]} == {"MaxPSI", "NetBenefit", "NetBenefitDropPct"}
