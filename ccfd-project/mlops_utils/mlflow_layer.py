"""
The MLflow layer: runs, the model registry and the aliases.

  connect()          point MLflow at the tracking URI from config.py and use experiment ccfd-fraud-detection;
                     a local server that does not answer stops it at once (server_answers)
  lineage_tags()     where a model came from: git sha, the data md5 (from the DVC pointer with --dvc-repo),
                     the row count and the base project script
  log_base_run()       one run: the base project's metrics, the model (pickle format), the lineage tags and evaluation.json
  base_results()       evaluation.json in its one shape: the base project's results JSON, as the base project's save_metrics_json writes it
  register()         a run's model becomes a new version of ccfd-fraud-model
  set_alias(), get_alias(), load_version()
                     @champion (the model in production) and @candidate (the next one the gate checks)
  download_alias()   an alias's model as a MODEL_DIR folder (model_latest.pkl, metadata_latest.json,
                     evaluation.json), the folder the serving image reads at /opt/ml/model
  latest_run()       the newest run with a given tag (for example journey_stage=gridsearch)

The same code runs against the local server (http://127.0.0.1:7080) and the SageMaker MLflow App
(its ARN): only MLFLOW_TRACKING_URI changes (MLOps 07: export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN").

This layer writes the MLflow registry only. The SageMaker registry (ccfd-fraud-detector) is written with
boto3 in other files (MLOps 09 and the pipeline), never from here.
"""

import hashlib
import importlib.util
import json
import logging
import pickle
import subprocess
import tempfile
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from mlops_utils import config

EXPERIMENT = config.EXPERIMENT_NAME        # ccfd-fraud-detection
MODEL_NAME = config.REGISTERED_MODEL       # ccfd-fraud-model
CHAMPION = config.ALIAS_CHAMPION           # champion
CANDIDATE = config.ALIAS_CANDIDATE         # candidate
MODEL_FILES_DIR = "model_dir"              # run artifact folder with the 2 JSON files of the model.tar.gz layout
NOT_FOUND_CODES = ("RESOURCE_DOES_NOT_EXIST",   # the registered model does not exist yet
                   "INVALID_PARAMETER_VALUE")   # the model exists, the alias does not ("alias ... not found")
# evaluation.json has one shape everywhere: the base project's results JSON. These are the keys of the base project's save_metrics_json.
BASE_RESULTS_KEYS = ("model_info", "model_parameters", "technical_metrics", "business_metrics", "dataset_info")
BASE_BUSINESS_KEYS = ("net_benefit", "fraud_caught", "fraud_missed", "false_alarms", "true_negatives",
                    "total_saved", "total_cost")


# ---------------------------------------------------------------------------
# 1. Connect
# ---------------------------------------------------------------------------
def server_answers(uri: str, timeout: float = 3) -> bool:
    """True when an MLflow server answers 'OK' on <uri>/health within the timeout (for http and https URIs)."""
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as response:
            return response.read().decode("utf-8", "replace").strip() == "OK"
    except (OSError, ValueError):
        return False


def connect(tracking_uri: str | None = None) -> str:
    """Point MLflow at the tracking URI (default: config.py) and select the experiment. Returns the URI."""
    uri = tracking_uri or config.load().tracking_uri
    if uri.startswith(("http://", "https://")) and not server_answers(uri):
        # Without this check MLflow retries for about 4 minutes (7 retries, backoff factor 2) before it fails
        raise RuntimeError(f"No MLflow server answers on {config.mask(uri)} (GET /health). "
                           "Start the local MLflow server on your EC2 VM (MLOps 04 Step-01-02), or, from MLOps 07 on, "
                           'point at your MLflow App: export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN"')
    if uri.startswith("arn:") and importlib.util.find_spec("sagemaker_mlflow") is None:
        raise RuntimeError("The tracking URI is an MLflow App ARN, so the sagemaker-mlflow package is needed "
                           "(pip install -r requirements-laptop.txt).")
    if not uri.startswith("arn:"):
        # sagemaker-mlflow 0.5.0 adds request headers for the App to EVERY MLflow call; with a non-ARN URI it
        # fails and MLflow prints one harmless WARNING per call ("... list index out of range"). Hide only that.
        logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    return uri


# ---------------------------------------------------------------------------
# 2. Lineage: where did this model come from?
# ---------------------------------------------------------------------------
def _git(folder: str | Path, *args: str):
    """Run one git command in folder. None when git is not installed (the slim training image has no git)."""
    try:
        return subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, check=False)
    except OSError:
        return None


def git_sha(folder: str | Path = ".") -> str:
    """The git commit of the code folder, or 'not-a-git-repo' (also when git is not installed)."""
    result = _git(folder, "rev-parse", "HEAD")
    return result.stdout.strip() if result is not None and result.returncode == 0 else "not-a-git-repo"


def git_dirty(folder: str | Path = ".") -> bool:
    """True when the code folder has changes that are not committed (then the git sha alone is not enough)."""
    result = _git(folder, "status", "--porcelain", "--", ".")
    return result is not None and result.returncode == 0 and result.stdout.strip() != ""


def file_md5(path: str | Path) -> str:
    """The md5 of a file (DVC 3 uses this same md5 for a single file)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dvc_md5(dvc_repo: str | Path, data_file: str) -> str:
    """
    The md5 that DVC recorded for data_file, read from its pointer file <dvc_repo>/<data_file>.dvc.
    It also checks that the data file in the workspace still has that md5 (else the tag would be wrong).
    """
    pointer = Path(dvc_repo) / f"{data_file}.dvc"
    if not pointer.is_file():
        raise FileNotFoundError(f"No DVC pointer {pointer}. Run 'dvc add {data_file}' in the lab repo first.")
    md5 = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
    workspace_file = Path(dvc_repo) / data_file
    if workspace_file.is_file() and file_md5(workspace_file) != md5:
        raise ValueError(f"{workspace_file} changed after 'dvc add': run 'dvc add' again or 'dvc checkout'.")
    return md5


def lineage_tags(base_script: str, data_file: str, dvc_repo: str | Path | None = None,
                 code_folder: str | Path = ".") -> dict[str, str]:
    """
    The lineage tags of a run:
      base_script    the base project script whose steps made the model (for example p5_03_train_production_model.py)
      data_md5     from the DVC pointer when dvc_repo is given (data_source=dvc), else the file's own md5
      data_file, data_rows, git_sha, git_dirty
    Student scripts pass dvc_repo (their --dvc-repo, default config.LAB_REPO). dvc_repo=None
    (data_source=file) is only for the checks and the tests, where no lab repo exists.
    """
    if dvc_repo is not None:
        data_path = Path(dvc_repo) / data_file
        md5, source = dvc_md5(dvc_repo, data_file), "dvc"
    else:
        data_path = Path(code_folder) / data_file
        md5, source = file_md5(data_path), "file"
    with open(data_path, encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1                  # minus the header line
    return {
        "base_script": base_script,
        "data_file": Path(data_file).name,
        "data_md5": md5,
        "data_source": source,
        "data_rows": str(rows),
        "git_sha": git_sha(code_folder),
        "git_dirty": str(git_dirty(code_folder)).lower(),
    }


# ---------------------------------------------------------------------------
# 3. Log one run
# ---------------------------------------------------------------------------
def is_base_results(content) -> bool:
    """True when content has the one evaluation.json shape: the base project's results JSON with business_metrics."""
    return (isinstance(content, dict) and isinstance(content.get("business_metrics"), dict)
            and "net_benefit" in content["business_metrics"])


def base_results(metrics: dict, model_name: str, params: dict | None = None,
               dataset_info: dict | None = None) -> dict:
    """
    evaluation.json for a model that has no base project results file (for example a model scored with gate.score()):
    the same 5 keys as the base project's save_metrics_json, filled from the numbers. Business numbers are whole dollars.
    """
    def first(*names):
        return next((float(metrics[name]) for name in names if metrics.get(name) is not None), None)

    now = datetime.now()
    return {
        "model_info": {"algorithm_name": model_name, "timestamp": now.isoformat(),
                       "run_id": f"{model_name.lower().replace(' ', '_')}_{now:%Y%m%d_%H%M%S}"},
        "model_parameters": params or {},
        "technical_metrics": {"f1_score": first("f1_score", "f1"), "precision": first("precision"),
                              "recall": first("recall"), "roc_auc": first("roc_auc")},
        "business_metrics": {key: int(metrics[key]) for key in BASE_BUSINESS_KEYS if metrics.get(key) is not None},
        "dataset_info": {"version": model_name, **(dataset_info or {})},
    }


def log_base_run(run_name: str, model, example, metrics: dict, tags: dict,
               params: dict | None = None, metadata: dict | None = None, evaluation: dict | None = None) -> str:
    """
    Log one trained base project model as an MLflow run and return its run id.

    model       the fitted sklearn Pipeline (logged in pickle format, the format the base project uses)
    example     a few input rows (a DataFrame without is_fraud) for the model signature
    metrics     the base project's numbers, for example from gate.score(); only numbers are logged
    tags        lineage_tags() plus anything else (for example journey_stage)
    params      the model's parameters (default: the Pipeline's 'model' step)
    metadata    the base project's metadata_latest.json for this model, if there is one (kept for MODEL_DIR)
    evaluation  the base project's results JSON for this model, if there is one (a training job passes p5_03's file); it is
                stored as evaluation.json exactly as it is. Without it, base_results() builds one from metrics.
    """
    if evaluation is not None and not is_base_results(evaluation):
        raise ValueError("evaluation must be the base project's results JSON (business_metrics with net_benefit), the one "
                         "evaluation.json shape.")
    if params is None and hasattr(model, "named_steps"):
        params = model.named_steps["model"].get_params()
    if hasattr(model, "feature_names_in_"):
        example = example[list(model.feature_names_in_)]   # the model's own inputs: 8 for p3_01, 9 with FE1
    numbers = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params or {})
        mlflow.log_metrics(numbers)
        with warnings.catch_warnings():
            # MLflow hints that integer columns cannot hold missing values; the base project's inputs never have any
            # (the serving API refuses a transaction with a missing field), so the hint is only noise here
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")
            mlflow.sklearn.log_model(
                model, name="model", serialization_format="pickle",
                signature=infer_signature(example, model.predict(example)), input_example=example.head(3),
            )
        # The 2 JSON files of the model.tar.gz layout, so download_alias() can rebuild the exact MODEL_DIR. The lineage
        # stays in the run's tags; evaluation.json keeps the base project's shape, the same as in every model.tar.gz
        model_type = type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__
        meta = metadata or {"model_name": run_name, "model_type": model_type, "metrics": numbers}
        if evaluation is None:
            evaluation = base_results(numbers, meta.get("model_name", run_name), params,
                                    {"data_source": tags.get("data_file"), "data_md5": tags.get("data_md5")})
        mlflow.log_dict(meta, f"{MODEL_FILES_DIR}/metadata_latest.json")
        mlflow.log_dict(evaluation, f"{MODEL_FILES_DIR}/evaluation.json")
    return run.info.run_id


def latest_run(tag: str, value: str):
    """The newest run in the experiment with tag=value, or None."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return None
    runs = MlflowClient().search_runs([experiment.experiment_id], filter_string=f"tags.{tag} = '{value}'",
                                      order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# 4. The registry: versions and aliases (MLflow only)
# ---------------------------------------------------------------------------
def register(run_id: str, tags: dict | None = None):
    """Register the model of a run as a new version of ccfd-fraud-model. Returns the ModelVersion."""
    client = MlflowClient()
    run = client.get_run(run_id)
    try:
        # MLflow 3 gives each logged model its own id; registering by that id is the clean way
        logged = client.search_logged_models(experiment_ids=[run.info.experiment_id],
                                             filter_string=f"source_run_id = '{run_id}'")
    except MlflowException:
        logged = []
    model_uri = f"models:/{logged[0].model_id}" if logged else f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri, MODEL_NAME)
    for key, value in {"source_run_id": run_id, **(tags or {})}.items():
        client.set_model_version_tag(MODEL_NAME, version.version, key, str(value))
    return version


def set_alias(alias: str, version: str | int) -> str | None:
    """Point an alias (champion or candidate) at a version. Returns the version it pointed at before, or None."""
    before = get_alias(alias)
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, str(version))
    return before.version if before else None


def get_alias(alias: str):
    """
    The ModelVersion an alias points at, or None if the alias (or the model) does not exist yet.
    Any other error (no access, a server error) is raised: it must never look like "no champion yet".
    """
    try:
        return MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException as error:
        if error.error_code in NOT_FOUND_CODES:
            return None
        raise


def tag_version(version: str | int, tags: dict) -> None:
    """Add tags to one model version (for example the gate result)."""
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, str(version), key, str(value))


def load_version(version: str | int):
    """Load one registered version as the sklearn Pipeline."""
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


# ---------------------------------------------------------------------------
# 5. An alias as a MODEL_DIR folder
# ---------------------------------------------------------------------------
def download_alias(alias: str, dest: str | Path):
    """
    Write the model an alias points at into dest, in the model.tar.gz layout that the serving image reads:
      model_latest.pkl       the Pipeline, pickled the way the base project's save_pipeline does it
      metadata_latest.json   the base project's metadata (+ an "mlflow" block: model, version, alias, run id)
      evaluation.json        the base project's results JSON as the run stored it; for a run without one, built from
                             the run's metrics with base_results()
    The serving image has no MLflow inside: this runs where MLflow answers (your EC2 VM, or CI), then MODEL_DIR=dest.
    Returns the ModelVersion.
    """
    version = get_alias(alias)
    if version is None:
        raise LookupError(f"No @{alias} alias on {MODEL_NAME} yet.")
    model = load_version(version.version)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "model_latest.pkl", "wb") as f:
        pickle.dump(model, f)

    run = MlflowClient().get_run(version.run_id)
    files = _run_json_files(version.run_id)
    metadata = files.get("metadata_latest.json") or {
        "model_name": run.info.run_name, "model_type": type(model).__name__, "metrics": run.data.metrics}
    metadata["mlflow"] = {"registered_model": MODEL_NAME, "version": version.version, "alias": alias,
                          "run_id": version.run_id}
    evaluation = files.get("evaluation.json")
    if not is_base_results(evaluation):
        evaluation = base_results(run.data.metrics, metadata.get("model_name") or run.info.run_name)
    for name, content in (("metadata_latest.json", metadata), ("evaluation.json", evaluation)):
        (dest / name).write_text(json.dumps(content, indent=2, default=str) + "\n", encoding="utf-8")
    return version


def _run_json_files(run_id: str) -> dict:
    """The JSON files log_base_run() stored in the run's model_dir folder (an empty dict if there are none)."""
    names = [Path(item.path).name for item in MlflowClient().list_artifacts(run_id, MODEL_FILES_DIR)]
    files = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{MODEL_FILES_DIR}/{name}",
                                                        dst_path=tmp)
            files[name] = json.loads(Path(local).read_text(encoding="utf-8"))
    return files
