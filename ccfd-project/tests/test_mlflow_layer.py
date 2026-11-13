"""
Tests for mlops_utils/mlflow_layer.py.

MLflow runs here on a SQLite file in a temporary folder: no server, no network, no AWS.
The model is a real the base project Pipeline (the base project's create_pipeline) trained on 200 made-up rows in a few milliseconds.
"""

import json
import os
import pickle
import socket
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
)

from mlops_utils import gate, mlflow_layer
from utils.data_utils import BASE_CATEGORICAL_COLS, BASE_NUMERICAL_COLS
from utils.model_utils import create_pipeline, load_pipeline

DATA_FILE = "data/credit_card_transactions_latest.csv"


def made_up_transactions(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "transaction_amount": rng.uniform(1, 2000, rows),
        "transaction_hour": rng.integers(0, 24, rows),
        "days_since_last_txn": rng.uniform(0, 30, rows),
        "avg_transaction_amount": rng.uniform(10, 500, rows),
        "merchant_category": rng.choice(["grocery", "electronics", "jewelry"], rows),
        "card_present": rng.integers(0, 2, rows),
        "international": rng.integers(0, 2, rows),
        "transaction_count_24h": rng.integers(0, 20, rows),
    })
    data["is_fraud"] = (data["transaction_amount"] > 1500).astype(int)
    return data


def base_pipeline(data, C):
    pipeline = create_pipeline(BASE_CATEGORICAL_COLS, BASE_NUMERICAL_COLS, model_params={"C": C})
    return pipeline.fit(data.drop(columns=["is_fraud"]), data["is_fraud"])


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """A temporary folder with a SQLite MLflow store and a data file; MLflow is reset afterwards."""
    folder = tmp_path_factory.mktemp("mlflow_layer")
    (folder / "data").mkdir()
    data = made_up_transactions()
    data.to_csv(folder / DATA_FILE, index=False)
    old_cwd, old_uri = Path.cwd(), mlflow.get_tracking_uri()
    os.chdir(folder)                                    # run artifacts go to <folder>/mlruns
    uri = mlflow_layer.connect(f"sqlite:///{folder / 'mlflow.db'}")
    yield {"folder": folder, "data": data, "uri": uri}
    os.chdir(old_cwd)
    mlflow.set_tracking_uri(old_uri)


def log_one(lab, run_name, C, stage):
    data = lab["data"]
    model = base_pipeline(data, C)
    tags = {**mlflow_layer.lineage_tags("p3_01_train_model_baseline.py", DATA_FILE, code_folder=lab["folder"]),
            "journey_stage": stage}
    metrics = gate.score(model, data)
    return model, mlflow_layer.log_base_run(run_name, model, data.drop(columns=["is_fraud"]), metrics, tags)


def test_connect_selects_the_experiment(lab):
    assert lab["uri"].startswith("sqlite:///")
    assert mlflow.get_experiment_by_name("ccfd-fraud-detection") is not None


def test_lineage_tags_from_the_file(lab):
    tags = mlflow_layer.lineage_tags("p5_03_train_production_model.py", DATA_FILE, code_folder=lab["folder"])
    assert tags["base_script"] == "p5_03_train_production_model.py"
    assert tags["data_rows"] == "200"
    assert tags["data_source"] == "file"
    assert tags["data_md5"] == mlflow_layer.file_md5(lab["folder"] / DATA_FILE)
    assert tags["git_sha"] == "not-a-git-repo"          # the temporary folder is not a git repo


def test_git_tags_without_git_installed(lab, monkeypatch):
    """The slim training image has no git: the lineage tags must still work (not-a-git-repo, not dirty)."""
    monkeypatch.setenv("PATH", "/nonexistent")
    assert mlflow_layer.git_sha(lab["folder"]) == "not-a-git-repo"
    assert mlflow_layer.git_dirty(lab["folder"]) is False
    tags = mlflow_layer.lineage_tags("p5_03_train_production_model.py", DATA_FILE, code_folder=lab["folder"])
    assert (tags["git_sha"], tags["git_dirty"]) == ("not-a-git-repo", "false")


def test_lineage_tags_from_a_dvc_pointer(lab, tmp_path):
    repo = tmp_path / "lab-repo"
    (repo / "data").mkdir(parents=True)
    lab["data"].to_csv(repo / DATA_FILE, index=False)
    md5 = mlflow_layer.file_md5(repo / DATA_FILE)
    pointer = repo / f"{DATA_FILE}.dvc"
    pointer.write_text(f"outs:\n- md5: {md5}\n  size: 1\n  hash: md5\n  path: credit_card_transactions_latest.csv\n",
                       encoding="utf-8")
    tags = mlflow_layer.lineage_tags("p3_01_train_model_baseline.py", DATA_FILE, dvc_repo=repo)
    assert (tags["data_md5"], tags["data_source"], tags["data_rows"]) == (md5, "dvc", "200")

    pointer.write_text(pointer.read_text(encoding="utf-8").replace(md5, "0" * 32), encoding="utf-8")
    with pytest.raises(ValueError):                     # the workspace file no longer matches DVC
        mlflow_layer.lineage_tags("p3_01_train_model_baseline.py", DATA_FILE, dvc_repo=repo)
    with pytest.raises(FileNotFoundError):              # no pointer at all
        mlflow_layer.dvc_md5(tmp_path / "no-repo", DATA_FILE)


def test_run_register_aliases_and_model_dir(lab, tmp_path):
    weak, weak_run = log_one(lab, "weak", C=0.001, stage="baseline")
    strong, strong_run = log_one(lab, "strong", C=10.0, stage="production")

    run = mlflow.get_run(strong_run)
    assert run.data.tags["journey_stage"] == "production" and run.data.tags["base_script"]
    assert "net_benefit" in run.data.metrics and "fraud_caught" in run.data.metrics

    v1 = mlflow_layer.register(weak_run, tags={"journey_stage": "baseline"})
    v2 = mlflow_layer.register(strong_run)
    assert (int(v2.version) - int(v1.version)) == 1
    assert mlflow.MlflowClient().get_model_version("ccfd-fraud-model", v1.version).tags["source_run_id"] == weak_run

    assert mlflow_layer.get_alias("candidate") is None
    assert mlflow_layer.set_alias("champion", v1.version) is None              # the first champion
    assert mlflow_layer.set_alias("champion", v2.version) == v1.version        # returns the old version
    assert mlflow_layer.get_alias("champion").version == v2.version

    model_dir = tmp_path / "champion"
    version = mlflow_layer.download_alias("champion", model_dir)
    assert version.version == v2.version
    assert sorted(path.name for path in model_dir.iterdir()) == \
        ["evaluation.json", "metadata_latest.json", "model_latest.pkl"]
    # evaluation.json has the one shape, the base project's results JSON, the same as in every model.tar.gz
    evaluation = json.loads((model_dir / "evaluation.json").read_text())
    assert tuple(evaluation) == mlflow_layer.BASE_RESULTS_KEYS
    assert evaluation["business_metrics"]["net_benefit"] == gate.score(strong, lab["data"])["net_benefit"]
    assert evaluation["dataset_info"]["version"] == "strong"
    X = lab["data"].drop(columns=["is_fraud"])
    with open(model_dir / "model_latest.pkl", "rb") as f:
        assert np.array_equal(pickle.load(f).predict_proba(X), strong.predict_proba(X))

    # The serving image loads MODEL_DIR with the base project's load_pipeline: it must read this folder as it is
    pipeline, metadata = load_pipeline(model_dir.name, save_dir=str(model_dir.parent))
    assert metadata["mlflow"] == {"registered_model": "ccfd-fraud-model", "version": v2.version,
                                  "alias": "champion", "run_id": strong_run}
    assert metadata["model_name"] == "strong" and metadata["model_type"] == "LogisticRegression"
    assert np.array_equal(pipeline.predict(X), strong.predict(X))

    # Rollback: move the alias back, download again, the folder now holds the old model
    assert mlflow_layer.set_alias("champion", v1.version) == v2.version
    mlflow_layer.download_alias("champion", model_dir)
    with open(model_dir / "model_latest.pkl", "rb") as f:
        assert np.array_equal(pickle.load(f).predict_proba(X), weak.predict_proba(X))


def test_a_base_results_json_is_stored_as_it_is(lab, tmp_path):
    """A training job passes p5_03's results JSON: download_alias() gives back exactly that file."""
    data = lab["data"]
    model = base_pipeline(data, 1.0)
    metrics = gate.score(model, data)
    base_file = mlflow_layer.base_results(metrics, "p5_03_production", dataset_info={"train_size": 160})
    tags = mlflow_layer.lineage_tags("p5_03_train_production_model.py", DATA_FILE, code_folder=lab["folder"])
    run_id = mlflow_layer.log_base_run("from-a-job", model, data.drop(columns=["is_fraud"]), metrics, tags,
                                     evaluation=base_file)
    mlflow_layer.set_alias("candidate", mlflow_layer.register(run_id).version)
    mlflow_layer.download_alias("candidate", tmp_path / "candidate")
    assert json.loads((tmp_path / "candidate" / "evaluation.json").read_text()) == base_file
    with pytest.raises(ValueError, match="one evaluation.json shape"):
        mlflow_layer.log_base_run("bad", model, data.drop(columns=["is_fraud"]), metrics, tags,
                                evaluation={"metrics": metrics, "lineage": tags})


def test_latest_run_by_tag(lab):
    log_one(lab, "gridsearch-1", C=1.0, stage="gridsearch")
    _, second = log_one(lab, "gridsearch-2", C=2.0, stage="gridsearch")
    assert mlflow_layer.latest_run("journey_stage", "gridsearch").info.run_id == second
    assert mlflow_layer.latest_run("journey_stage", "never-used") is None


def test_download_without_alias_fails(lab, tmp_path):
    with pytest.raises(LookupError):
        mlflow_layer.download_alias("no-such-alias", tmp_path / "x")


def test_connect_fails_fast_without_a_server():
    """No server on the port: a clear error within seconds (MLflow alone retries for about 4 minutes)."""
    with socket.socket() as probe:                       # a free local port that nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    uri_before = mlflow.get_tracking_uri()
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="No MLflow server answers on http://127.0.0.1:") as error:
        mlflow_layer.connect(f"http://127.0.0.1:{port}")
    assert time.monotonic() - start < 10
    # The hint is true in every section folder: no script path (the start script is only in MLOps 04 and 05)
    assert "MLOps 04 Step-01-02" in str(error.value)
    assert 'export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN"' in str(error.value)
    assert "m04_01_start_mlflow.sh" not in str(error.value)
    assert mlflow.get_tracking_uri() == uri_before        # nothing was changed


class FakeClient:
    """Stands in for MlflowClient: get_model_version_by_alias raises the MLflow error it was given."""

    def __init__(self, error_code):
        self.error_code = error_code

    def get_model_version_by_alias(self, name, alias):
        raise MlflowException(f"fake {alias}", error_code=self.error_code)


def test_get_alias_none_only_when_not_found(monkeypatch):
    """Only 'model or alias not found' means 'no alias yet'; no access or a server error must come through."""
    for code in (RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE):
        monkeypatch.setattr(mlflow_layer, "MlflowClient", lambda code=code: FakeClient(code))
        assert mlflow_layer.get_alias("champion") is None
    monkeypatch.setattr(mlflow_layer, "MlflowClient", lambda: FakeClient(PERMISSION_DENIED))
    with pytest.raises(MlflowException) as error:
        mlflow_layer.get_alias("champion")
    assert error.value.error_code == "PERMISSION_DENIED"
    with pytest.raises(MlflowException):                 # download_alias never says "no alias yet" for it
        mlflow_layer.download_alias("champion", "unused")


def test_layer_never_writes_the_sagemaker_registry():
    """the SageMaker registry ccfd-fraud-detector is written by boto3 elsewhere, never by this layer."""
    source = Path(mlflow_layer.__file__).read_text(encoding="utf-8")
    assert "import boto3" not in source and "create_model_package" not in source
