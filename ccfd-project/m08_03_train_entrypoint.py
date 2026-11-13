"""
MLOps 08: the SageMaker training entrypoint around the base project's p5_02 and p5_03.

SageMaker starts a training container as "docker run <image> train"; the train launcher runs this file.

What SageMaker gives the container:
  /opt/ml/input/data/fe/<one file>              channel fe: the base project's FE1 data (any name, no extension needed)
  /opt/ml/input/data/config/<one file>          channel config: the base project's fixed YAML (stage production)
  /opt/ml/input/config/hyperparameters.json     {"stage": "production"} (every SageMaker job) or "gridsearch"
What this file writes:
  /opt/ml/model/model_latest.pkl, metadata_latest.json, evaluation.json
                                                the model.tar.gz layout; SageMaker packs this folder into model.tar.gz
  /opt/ml/output/data/p5_02_best_config.yaml    the YAML the run used (SageMaker packs it into output.tar.gz)
  /opt/ml/output/failure                        the reason, when the run fails (SageMaker shows it as FailureReason)

The 2 stages (hyperparameter "stage"; without it: production):
  production   every SageMaker job (MLOps 08 and 09, the pipeline). The YAML from channel config is
               copied to artifacts/p5_02_best_config.yaml, then only the base project's p5_03 runs (about 1 minute).
  gridsearch   the laptop and a local container only. The base project's p5_02 (the 300-fit grid search, about 17 minutes on
               ml.m5.large) writes the YAML, then p5_03 runs. On Linux p5_02 picks C 10 (test Net Benefit 47,925);
               The base project's macOS run picked C 0.1 (50,550), and that YAML is the fixed config of every SageMaker job.

The base project's scripts run unchanged as separate Python processes in a work copy under /tmp: they read and write
data/, artifacts/, models/ and results/ next to themselves, so a run never touches this folder's own files.

MLflow: when MLFLOW_TRACKING_URI is set (in a SageMaker job: the MLflow App ARN, passed in the job's
Environment), the run is logged with mlops_utils/mlflow_layer.py. When it is not set, MLflow is not even imported.

On the laptop (or in a test), point it at a folder in the SageMaker layout:
    python m08_03_train_entrypoint.py --opt-ml sm-local
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

# mlops_utils/config.py uses the standard library only; config.mask() hides a 12-digit account ID
from mlops_utils import config

HERE = Path(__file__).resolve().parent

# The base project's own file names (the scripts read and write exactly these, relative to their folder)
FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"
CONFIG_FILE = "artifacts/p5_02_best_config.yaml"
GRIDSEARCH_SCRIPT = "p5_02_class_weight_gridsearch.py"
PRODUCTION_SCRIPT = "p5_03_train_production_model.py"
PRODUCTION_MODEL = "models/p5_03_production"
PRODUCTION_RESULTS = "*p5_03_production*.json"       # the base project's save_metrics_json file of p5_03, in results/
MODEL_FILES = ("model_latest.pkl", "metadata_latest.json", "evaluation.json")
STAGES = ("production", "gridsearch")
FAILURE_LIMIT = 1024                                  # SageMaker keeps the first 1,024 characters as FailureReason


class TrainError(Exception):
    """A problem with the job's inputs: the message is written to /opt/ml/output/failure as it is."""


def say(message: str) -> None:
    print(f"[m08_03] {message}", flush=True)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. Read the job's inputs
# ---------------------------------------------------------------------------
def single_file(channel: Path) -> Path:
    """The one file in a channel folder, whatever its name. Hidden files such as .DS_Store are skipped."""
    if not channel.is_dir():
        raise TrainError(f"No channel folder {channel}.")
    files = sorted(path for path in channel.rglob("*") if path.is_file() and not path.name.startswith("."))
    if len(files) != 1:
        names = ", ".join(str(path.relative_to(channel)) for path in files[:5]) or "none"
        raise TrainError(f"Channel {channel.name} must hold exactly 1 file, found {len(files)} ({names}).")
    return files[0]


def read_stage(opt_ml: Path) -> str:
    """The hyperparameter "stage" from input/config/hyperparameters.json; production when there is none."""
    path = opt_ml / "input" / "config" / "hyperparameters.json"
    stage = "production"
    if path.is_file():
        stage = str(json.loads(path.read_text(encoding="utf-8")).get("stage", stage)).strip().lower()
    if stage not in STAGES:
        raise TrainError(f"Unknown stage '{stage}'. Use one of: {', '.join(STAGES)}.")
    return stage


def read_config(path: Path) -> dict:
    """The base project's p5_02 YAML; p5_03 needs its best_params."""
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise TrainError(f"Channel config: {path.name} is not valid YAML ({error}).") from error
    if not isinstance(content, dict) or not isinstance(content.get("best_params"), dict):
        raise TrainError(f"Channel config: {path.name} has no best_params (expected the base project's p5_02_best_config.yaml).")
    return content


# ---------------------------------------------------------------------------
# 2. Run the base project's scripts in a work copy
# ---------------------------------------------------------------------------
def make_work_copy(code_dir: Path, work: Path) -> None:
    """Copy the base project's utils/, p5_02 and p5_03 (unchanged) into the work folder; data/ and artifacts/ start empty."""
    shutil.copytree(code_dir / "utils", work / "utils", ignore=shutil.ignore_patterns("__pycache__"))
    for script in (GRIDSEARCH_SCRIPT, PRODUCTION_SCRIPT):
        shutil.copy2(code_dir / script, work / script)
    (work / "data").mkdir()
    (work / "artifacts").mkdir()


def run_base(script: str, work: Path) -> float:
    """Run one base project script in the work folder, exactly as the base project runs it (python <script>). Returns the seconds."""
    say(f"=== The base project {script}")
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=work, check=False)
    if result.returncode != 0:
        raise TrainError(f"The base project's {script} failed (exit code {result.returncode}); its own output is in the log above.")
    seconds = time.time() - start
    say(f"=== The base project {script} done in {seconds:.1f} s")
    return seconds


def write_model_dir(work: Path, model_dir: Path) -> dict:
    """The model.tar.gz layout at the model folder root: p5_03's model and metadata, and its results JSON as evaluation.json."""
    results = sorted((work / "results").glob(PRODUCTION_RESULTS))
    if len(results) != 1:
        raise TrainError(f"Expected 1 p5_03 results file in results/, found {len(results)}.")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file, metadata_file, evaluation_file = MODEL_FILES
    for name in (model_file, metadata_file):
        shutil.copy2(work / PRODUCTION_MODEL / name, model_dir / name)
    shutil.copy2(results[0], model_dir / evaluation_file)          # the base project's results JSON: the one shape
    return json.loads((model_dir / evaluation_file).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 3. MLflow, only when MLFLOW_TRACKING_URI is set
# ---------------------------------------------------------------------------
def log_to_mlflow(work: Path, model_dir: Path, stage: str, evaluation: dict, config_md5: str) -> str | None:
    """Log the run with mlops_utils/mlflow_layer.py. Returns the run id, or None when MLFLOW_TRACKING_URI is not set."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        say("MLflow: MLFLOW_TRACKING_URI is not set, so nothing is logged (the model is in /opt/ml/model)")
        return None
    # MLflow is imported only now, when a run is really logged
    import pickle

    import pandas as pd

    from mlops_utils import mlflow_layer

    mlflow_layer.connect(uri)
    with open(model_dir / "model_latest.pkl", "rb") as f:
        model = pickle.load(f)
    example = pd.read_csv(work / FE_FILE, nrows=5).drop(columns=["is_fraud"])
    metrics = {**evaluation.get("technical_metrics", {}), **evaluation.get("business_metrics", {})}
    metadata = json.loads((model_dir / "metadata_latest.json").read_text(encoding="utf-8"))
    job_name = os.environ.get("TRAINING_JOB_NAME", "local")    # SageMaker sets TRAINING_JOB_NAME in a job
    tags = mlflow_layer.lineage_tags(PRODUCTION_SCRIPT, FE_FILE, dvc_repo=None, code_folder=work)
    tags.update({"journey_stage": stage, "runtime": "sagemaker" if job_name != "local" else "local",
                 "training_job": job_name, "data_source": "channel-fe", "config_md5": config_md5})
    run_id = mlflow_layer.log_base_run(f"m08-{stage}-{job_name}", model, example, metrics, tags, metadata=metadata,
                                     evaluation=evaluation)
    say(config.mask(f"MLflow: run {run_id} logged to {uri} (experiment {mlflow_layer.EXPERIMENT})"))
    return run_id


# ---------------------------------------------------------------------------
# 4. The job
# ---------------------------------------------------------------------------
def train(opt_ml: Path, code_dir: Path = HERE) -> dict:
    """Run one training job in the SageMaker layout under opt_ml. Returns a summary; raises on any problem."""
    stage = read_stage(opt_ml)
    fe_input = single_file(opt_ml / "input" / "data" / "fe")
    config_channel = opt_ml / "input" / "data" / "config"
    model_dir = opt_ml / "model"
    output_data = opt_ml / "output" / "data"
    say(f"stage {stage}; channel fe: {fe_input.name} ({fe_input.stat().st_size:,} bytes, md5 {md5(fe_input)})")
    config_input = None
    if stage == "production":
        # Check every input before the base project starts: a bad job fails in a second, with a clear reason
        if not config_channel.is_dir():
            raise TrainError("Stage production needs channel config with the base project's fixed YAML "
                             "(p5_02_best_config.yaml), and the job has no channel config.")
        config_input = single_file(config_channel)
        read_config(config_input)
    elif config_channel.is_dir():
        say("stage gridsearch: channel config is not used (p5_02 writes the YAML)")

    with tempfile.TemporaryDirectory(prefix="base-work-") as tmp:
        work = Path(tmp)
        make_work_copy(code_dir, work)
        shutil.copy2(fe_input, work / FE_FILE)
        say(f"channel fe -> {FE_FILE}")
        seconds = {}
        if config_input is not None:
            shutil.copy2(config_input, work / CONFIG_FILE)
            say(f"channel config: {config_input.name} (md5 {md5(config_input)}) -> {CONFIG_FILE}; p5_02 skipped")
        else:
            seconds[GRIDSEARCH_SCRIPT] = run_base(GRIDSEARCH_SCRIPT, work)
        seconds[PRODUCTION_SCRIPT] = run_base(PRODUCTION_SCRIPT, work)

        evaluation = write_model_dir(work, model_dir)
        output_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / CONFIG_FILE, output_data / "p5_02_best_config.yaml")
        used = read_config(work / CONFIG_FILE)
        config_md5 = md5(work / CONFIG_FILE)
        run_id = log_to_mlflow(work, model_dir, stage, evaluation, config_md5)

    business = evaluation.get("business_metrics", {})
    summary = {
        "stage": stage,
        "best_params": used["best_params"],
        "best_cv_score": used.get("best_cv_score"),
        "config_test_net_benefit": used.get("test_net_benefit"),
        "net_benefit": business.get("net_benefit"),
        "config_md5": config_md5,
        "seconds": {script: round(value, 1) for script, value in seconds.items()},
        "model_dir": sorted(path.name for path in model_dir.iterdir()),
        "mlflow_run_id": run_id,
    }
    say("TRAIN SUMMARY " + json.dumps(summary, default=str))
    return summary


def write_failure(opt_ml: Path, message: str) -> None:
    """SageMaker reads /opt/ml/output/failure and shows its text as the job's FailureReason."""
    try:
        failure = opt_ml / "output" / "failure"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(message[:FAILURE_LIMIT], encoding="utf-8")
    except OSError as error:
        print(f"[m08_03] could not write the failure file: {error}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training entrypoint (SageMaker training contract).")
    parser.add_argument("--opt-ml", type=Path, default=Path("/opt/ml"),
                        help="the SageMaker folder (default /opt/ml; on the laptop for example sm-local)")
    parser.add_argument("--code-dir", type=Path, default=HERE,
                        help="where the base project's utils/, p5_02 and p5_03 are (default: this file's folder)")
    args = parser.parse_args(argv)
    # Every failure ends in the failure file and a non-zero exit code (SageMaker: Failed, with that reason)
    try:
        train(args.opt_ml, args.code_dir)
    except TrainError as error:
        message = config.mask(str(error))
        write_failure(args.opt_ml, message)
        say(f"FAILED: {message}")
        return 1
    except Exception as error:  # noqa: BLE001 (the top-level handler: every failure must reach the failure file)
        message = config.mask(f"{type(error).__name__}: {error}")
        write_failure(args.opt_ml, message)
        # The traceback goes to the job log (CloudWatch on AWS), so it is masked too: never a bare raise
        say(f"FAILED: {message} (the masked traceback follows)")
        print(config.mask(traceback.format_exc()), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
