"""MLOps 05: the registry and alias rules the m05 scripts use (a SQLite tracking store in a temp folder)."""
import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlops_utils import gate, mlflow_layer


@pytest.fixture
def store(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(mlflow_layer.EXPERIMENT)
    yield
    mlflow.set_tracking_uri("")


def logged_run(stage: str) -> str:
    model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    with mlflow.start_run(run_name=stage) as run:
        mlflow.set_tag("journey_stage", stage)
        mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
    return run.info.run_id


def test_versions_are_kept_and_the_alias_moves_both_ways(store):
    v1 = mlflow_layer.register(logged_run("fe1"), {"gate_result": "FIRST_CHAMPION"})
    assert mlflow_layer.get_alias(mlflow_layer.CHAMPION) is None
    assert mlflow_layer.set_alias(mlflow_layer.CHAMPION, v1.version) is None
    v2 = mlflow_layer.register(logged_run("production"))
    assert mlflow_layer.set_alias(mlflow_layer.CHAMPION, v2.version) == v1.version
    assert mlflow_layer.set_alias(mlflow_layer.CHAMPION, v1.version) == v2.version   # roll back
    assert mlflow_layer.get_alias(mlflow_layer.CHAMPION).version == v1.version
    versions = mlflow.MlflowClient().search_model_versions(f"name = '{mlflow_layer.MODEL_NAME}'")
    assert len(versions) == 2


def test_a_rejected_candidate_keeps_its_alias_and_the_champion_stays(store):
    v1 = mlflow_layer.register(logged_run("production"))
    mlflow_layer.set_alias(mlflow_layer.CHAMPION, v1.version)
    v2 = mlflow_layer.register(logged_run("randomsearch"))
    mlflow_layer.set_alias(mlflow_layer.CANDIDATE, v2.version)
    decision, promote = gate.decide(37025, 50550)
    assert (decision, promote) == ("REJECT", 0)
    mlflow_layer.tag_version(v2.version, {"gate_result": decision})
    assert mlflow_layer.get_alias(mlflow_layer.CHAMPION).version == v1.version
    assert mlflow_layer.get_alias(mlflow_layer.CANDIDATE).version == v2.version
    tags = mlflow.MlflowClient().get_model_version(mlflow_layer.MODEL_NAME, v2.version).tags
    assert tags["gate_result"] == "REJECT"


def test_a_tie_is_not_a_promotion():
    assert gate.decide(50550, 50550) == ("REJECT", 0)
