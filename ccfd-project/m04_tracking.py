"""
MLOps 04: the shared steps of the m04 scripts (track one base project script as an MLflow run).

  run_base(script)        runs the base project's script unchanged, in this process, and returns its variables
                        (the trained model, X_test, the search object, the base project's own metrics)
  data_tags(...)        lineage tags; the data md5 is checked against the lab repo's DVC pointer at a tag
  track(...)            one parent run: the base project's metrics on the base project's test split, the model (pickle), the lineage tags,
                        2 MLflow datasets (context "training" = the 8,000 training rows, context "test" = the 2,000
                        held-out test rows the run was scored on) and a confusion matrix picture
  log_children(...)     one nested child run per search candidate (params, cv_net_benefit, rank)

The tracking URI comes only from MLFLOW_TRACKING_URI (never written in code), so MLOps 07 reruns these same
scripts against the SageMaker MLflow App by changing that one variable.
"""

import argparse
import contextlib
import io
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from mlops_utils import config, gate, mlflow_layer

# The base project's split (80/20, stratified, seed 42), unchanged
from utils.data_utils import split_data

DATA_FILE = "data/credit_card_transactions_latest.csv"   # the file DVC versions (MLOps 03)


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dvc-repo", type=Path, default=config.LAB_REPO,
                        help="the lab repo of MLOps 03 (default ~/mlops-lab/ccfd-project)")
    parser.add_argument("--data-tag", default="data-v1", help="the data version this run must use (default data-v1)")
    return parser.parse_args()


def connect() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        sys.exit("STOP: MLFLOW_TRACKING_URI is not set. Run: export MLFLOW_TRACKING_URI=http://127.0.0.1:7080")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")   # no download bars in the output
    # A server that does not answer stops with one STOP line, never a traceback
    try:
        return mlflow_layer.connect(uri)
    except RuntimeError as error:
        sys.exit(f"STOP: {error}")


def run_base(script: str) -> dict:
    """Run the base project's script exactly as 'python <script>' would, and keep its variables."""
    print(f"Running the base project's {script} (unchanged) ...", flush=True)
    return runpy.run_path(script, run_name="__main__")


def pointer_md5_at_tag(dvc_repo: Path, tag: str) -> str:
    """The data md5 in the lab repo's .dvc pointer at a git tag (git show <tag>:<pointer>)."""
    shown = subprocess.run(["git", "-C", str(dvc_repo), "show", f"{tag}:{DATA_FILE}.dvc"],
                           check=False, capture_output=True, text=True)
    if shown.returncode != 0:
        sys.exit(f"STOP: no {DATA_FILE}.dvc at tag {tag} in {dvc_repo} (MLOps 03 Step-02 makes it).")
    return yaml.safe_load(shown.stdout)["outs"][0]["md5"]


def data_tags(base_script: str, stage: str, args: argparse.Namespace) -> dict:
    """Lineage tags; the data on disk must be the data version named by --data-tag in the lab repo."""
    tags = mlflow_layer.lineage_tags(base_script, DATA_FILE)
    want = pointer_md5_at_tag(args.dvc_repo.expanduser(), args.data_tag)
    if tags["data_md5"] != want:
        sys.exit(f"STOP: {DATA_FILE} has md5 {tags['data_md5']}, but {args.data_tag} in the lab repo is {want}.")
    print(f"Data: md5 {want} = {args.data_tag} in the lab repo")
    return {**tags, "data_source": "dvc", "data_version": args.data_tag, "journey_stage": stage}


def _confusion_matrix(metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [[metrics["true_negatives"], metrics["false_alarms"]], [metrics["fraud_missed"], metrics["fraud_caught"]]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cells, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cells[row][col]:,}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["legit", "fraud"])
    ax.set_yticks([0, 1], ["legit", "fraud"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def split_rows(train_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training rows and the test rows of a data file, split the way the base project's script split it."""
    with contextlib.redirect_stdout(io.StringIO()):   # split_data prints progress lines; keep our output short
        X_train, X_test, y_train, y_test, _ = split_data(pd.read_csv(train_csv))
    return X_train.assign(**{gate.TARGET: y_train}), X_test.assign(**{gate.TARGET: y_test})


def track(run_name: str, model, base_net_benefit: float, tags: dict, train_csv: str,
          params: dict | None = None) -> str:
    """Log the parent run and return its run id. Its Net Benefit must equal the number the base project's script printed."""
    test = gate.base_test_set()
    metrics = gate.score(model, test)
    if metrics["net_benefit"] != int(base_net_benefit):
        sys.exit(f"STOP: Net Benefit {metrics['net_benefit']} is not the base project's {int(base_net_benefit)}.")
    run_id = mlflow_layer.log_base_run(run_name, model, test.drop(columns=[gate.TARGET]), metrics, tags, params=params)
    with mlflow.start_run(run_id=run_id), tempfile.TemporaryDirectory() as folder:
        # 2 datasets, never the whole file as training: the rows the model trained on, and the held-out test rows
        # it was scored on (so the lineage shows the test rows were not trained on)
        training_rows, test_rows = split_rows(train_csv)
        for rows, context in ((training_rows, "training"), (test_rows, "test")):
            mlflow.log_input(mlflow.data.from_pandas(rows, source=train_csv, name=Path(train_csv).name,
                                                     targets=gate.TARGET), context=context)
        picture = Path(folder) / "confusion_matrix.png"
        _confusion_matrix(metrics, picture)
        mlflow.log_artifact(str(picture))
    print(f"Run {run_name}: Net Benefit ${metrics['net_benefit']:,} (= the base project's own number), F1 {metrics['f1']:.4f}, "
          f"caught {metrics['fraud_caught']}/{metrics['fraud_caught'] + metrics['fraud_missed']}, "
          f"false alarms {metrics['false_alarms']:,}")
    return run_id


def log_children(parent_run_id: str, search) -> int:
    """One nested child run per candidate of a fitted RandomizedSearchCV or GridSearchCV."""
    results = search.cv_results_
    with mlflow.start_run(run_id=parent_run_id):
        for i, candidate in enumerate(results["params"]):
            clean = {key.replace("model__", ""): str(value) for key, value in candidate.items()}
            with mlflow.start_run(run_name=f"candidate-{i + 1:02d}", nested=True):
                mlflow.log_params(clean)
                mlflow.log_metrics({"cv_net_benefit": float(results["mean_test_score"][i]),
                                    "rank": int(results["rank_test_score"][i])})
    return len(results["params"])
