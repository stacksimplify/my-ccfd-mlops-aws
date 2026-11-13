"""
Tests for mlops_utils/gate.py, the Net Benefit gate.

PROMOTE only if the candidate is strictly better; a tie keeps the champion; the 'promote' field is 1 or 0
(the pipeline's condition step reads it); no champion yet promotes the first candidate.
The small tests use made-up models with fixed probabilities, so every Net Benefit is known in advance:
  The base project's money rule: caught fraud +1,475, missed fraud -1,500, false alarm -25, correct legit 0.
The last test scores the base project's real models on the base project's test set (skipped if the Step-01 laptop models are missing;
with MLOPS_REQUIRE_MODEL=1 a missing file fails the test instead).
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_utils import gate

PROJECT = Path(__file__).resolve().parents[1]


class FixedModel:
    """A stand-in for a base project Pipeline: predict_proba returns the probabilities it was given."""

    def __init__(self, fraud_probabilities):
        self.p = np.asarray(fraud_probabilities, dtype=float)

    def predict_proba(self, X):
        assert len(X) == len(self.p)
        return np.column_stack([1 - self.p, self.p])


# 4 frauds and 6 legit transactions
LABELS = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
DATA = pd.DataFrame({"transaction_amount": np.arange(10.0), "is_fraud": LABELS})

# catches 3 of 4 frauds, 1 false alarm: 3 x 1,475 - 1,500 - 25 = 2,900
GOOD = FixedModel([0.9, 0.8, 0.7, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
# catches 2 of 4 frauds, 0 false alarms: 2 x 1,475 - 2 x 1,500 = -50
WEAK = FixedModel([0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def gate_run(candidate, champion, tmp_path=None):
    out = tmp_path / "gate_decision.json" if tmp_path else None
    return gate.run_gate(candidate, champion, DATA, candidate_name="ccfd-fraud-model", candidate_version="2",
                         champion_name="ccfd-fraud-model", champion_version="1", data_name="test data",
                         out_path=out)


def test_score_uses_base_money_rule():
    result = gate.score(GOOD, DATA)
    assert (result["fraud_caught"], result["fraud_missed"], result["false_alarms"]) == (3, 1, 1)
    assert result["net_benefit"] == 3 * 1475 - 1500 - 25 == 2900
    assert gate.score(WEAK, DATA)["net_benefit"] == -50


def test_probability_of_exactly_half_is_fraud():
    """The same rule as the serving API: FRAUD when the probability is 0.5 or more."""
    at_half = FixedModel([0.5] * 4 + [0.0] * 6)
    assert gate.score(at_half, DATA)["fraud_caught"] == 4


def test_promote_when_candidate_is_strictly_better(tmp_path):
    record = gate_run(GOOD, WEAK, tmp_path)
    assert record["decision"] == "PROMOTE"
    assert record["promote"] == 1
    assert record["net_benefit_gain"] == 2900 - (-50)
    assert (record["candidate"]["net_benefit"], record["champion"]["net_benefit"]) == (2900, -50)
    assert record["serve_model_version"] == "2"


def test_reject_when_candidate_is_worse(tmp_path):
    record = gate_run(WEAK, GOOD, tmp_path)
    assert record["decision"] == "REJECT"
    assert record["promote"] == 0
    assert record["net_benefit_gain"] == -2950
    assert record["serve_model_version"] == "1"          # the champion stays
    assert record["reason"] == "candidate -50 <= champion 2,900 (-2,950)"


def test_tie_keeps_the_champion(tmp_path):
    record = gate_run(GOOD, FixedModel(GOOD.p), tmp_path)
    assert record["net_benefit_gain"] == 0
    assert record["decision"] == "REJECT"
    assert record["promote"] == 0
    assert record["serve_model_version"] == "1"
    assert "tie" in record["reason"] and "tie" in record["notes"][0]


def test_no_champion_yet_promotes_the_first_candidate():
    record = gate.run_gate(WEAK, None, DATA, candidate_name="ccfd-fraud-model", candidate_version="1")
    assert (record["decision"], record["promote"], record["champion"], record["net_benefit_gain"]) == \
        ("PROMOTE", 1, None, None)


def test_promote_field_in_the_json_file(tmp_path):
    """The pipeline condition reads 'promote' from gate_decision.json: it must be the number 1 or 0."""
    for candidate, champion, expected in ((GOOD, WEAK, 1), (WEAK, GOOD, 0), (GOOD, GOOD, 0)):
        gate_run(candidate, champion, tmp_path)
        saved = json.loads((tmp_path / "gate_decision.json").read_text(encoding="utf-8"))
        assert saved["promote"] == expected and type(saved["promote"]) is int
        assert saved["decision"] == ("PROMOTE" if expected else "REJECT")
        for key in ("champion", "candidate", "net_benefit_gain", "decision", "promote", "serve_model_name",
                    "serve_model_version", "notes", "decided_at", "gate_rows", "gate_frauds", "rule"):
            assert key in saved
        assert saved["decided_at"].endswith("+00:00")                 # UTC, the same on the laptop and in a job


def test_decide_rule():
    assert gate.decide(10, 9) == ("PROMOTE", 1)
    assert gate.decide(9, 9) == ("REJECT", 0)
    assert gate.decide(8, 9) == ("REJECT", 0)
    assert gate.decide(-5, None) == ("PROMOTE", 1)


def test_version_tags():
    tags = gate.version_tags(gate_run(GOOD, WEAK))
    assert tags["gate_result"] == "PROMOTE"
    assert (tags["candidate_net_benefit"], tags["champion_net_benefit"], tags["net_benefit_gain"]) == \
        ("2900", "-50", "2950")
    first = gate.version_tags(gate.run_gate(WEAK, None, DATA, candidate_name="ccfd-fraud-model", candidate_version="1"))
    assert (first["champion_net_benefit"], first["net_benefit_gain"]) == ("none", "none")   # one spelling


def test_add_amount_ratio_only_when_missing():
    raw = pd.DataFrame({"transaction_amount": [100.0, 50.0], "avg_transaction_amount": [50.0, 0.0]})
    added = gate.add_amount_ratio(raw)
    assert list(added["amount_ratio"]) == [2.0, 50.0]      # the base project divides by 1 when the average is 0
    assert "amount_ratio" not in raw.columns                # the caller's DataFrame is not changed
    assert gate.add_amount_ratio(added) is added


BASE_MODELS = [PROJECT / "models" / name / "model_latest.pkl"
             for name in ("p3_01_baseline", "p5_01_randomsearch", "p5_03_production")]


def test_base_models_on_base_test_set():
    """The course story offline: p5_03 beats p3_01 (PROMOTE), p5_01 loses to p5_03 (REJECT), p5_03 ties itself."""
    missing = [str(path.relative_to(PROJECT)) for path in (*BASE_MODELS, PROJECT / gate.BASE_FE_FILE) if not path.exists()]
    if missing:
        message = f"needs the Step-01 laptop models and the base project's FE1 data (run the base project's chain p1_01 ... p5_03); missing: " \
                  f"{', '.join(missing)}"
        if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
            pytest.fail(f"MLOPS_REQUIRE_MODEL=1, but this test {message}")   # CI and the build checks never skip it
        pytest.skip(message)
    p3_01, p5_01, p5_03 = (pickle.loads(path.read_bytes()) for path in BASE_MODELS)
    data = gate.base_test_set(PROJECT / gate.BASE_FE_FILE)
    assert (len(data), int(data["is_fraud"].sum())) == (2000, 60)

    first = gate.run_gate(p5_03, p3_01, data, candidate_name="p5_03", candidate_version="2",
                          champion_name="p3_01", champion_version="1")
    assert (first["champion"]["net_benefit"], first["candidate"]["net_benefit"]) == (39625, 50550)
    assert (first["decision"], first["promote"], first["net_benefit_gain"]) == ("PROMOTE", 1, 10925)

    second = gate.run_gate(p5_01, p5_03, data, candidate_name="p5_01", candidate_version="3",
                           champion_name="p5_03", champion_version="2")
    assert second["candidate"]["net_benefit"] == 37025
    assert (second["decision"], second["promote"], second["net_benefit_gain"]) == ("REJECT", 0, -13525)

    tie = gate.run_gate(p5_03, p5_03, data, candidate_name="p5_03", candidate_version="2",
                        champion_name="p5_03", champion_version="2")
    assert (tie["decision"], tie["promote"], tie["net_benefit_gain"]) == ("REJECT", 0, 0)
