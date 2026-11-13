"""
Tests for m08_03_train_entrypoint.py, the SageMaker training entrypoint.

Each test builds a fake /opt/ml folder in a temporary folder, the way SageMaker fills it:
  input/data/fe/<one file, no extension>     (here: the md5 of the file, like a DVC object)
  input/data/config/p5_02_best_config.yaml   (the base project's fixed YAML, config/p5_02_best_config.yaml)
  input/config/hyperparameters.json          {"stage": "production"}
and runs the entrypoint with --opt-ml <that folder>, as "python m08_03_train_entrypoint.py".

The production tests run the base project's real p5_03 (about 2 seconds) on the laptop FE1 data; they skip without it, unless
MLOPS_REQUIRE_MODEL=1 (see tests/conftest.py). The gridsearch tests use 2 small stand-in scripts instead of the base project's
p5_02 and p5_03, so they prove the order and the files in under a second (the real grid search is the local
container run of Step-07 T2). The last tests read Dockerfile.training and the train launcher: every module the 3
entrypoints import (and every base project script they run) must be copied into the image. With MLOPS_TRAINING_IMAGE set
(tests/conftest.py), the built image itself imports the 3 entrypoints and runs the train launcher.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT / "m08_03_train_entrypoint.py"
FE_DATA = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
BASE_CONFIG = PROJECT / "config" / "p5_02_best_config.yaml"
N09 = ["evaluation.json", "metadata_latest.json", "model_latest.pkl"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def need_fe_data():
    if not FE_DATA.exists():
        message = "needs the laptop FE1 data: run the base project's p1_01 and p4_01 first"
        if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
            pytest.fail(f"MLOPS_REQUIRE_MODEL=1, but this test {message}")
        pytest.skip(message)


def make_opt_ml(root: Path, stage: str | None = "production", fe_name: str | None = None,
                config: Path | None = BASE_CONFIG, fe_source: Path = FE_DATA) -> Path:
    """A SageMaker-shaped folder: channel fe (one extension-less file), channel config, hyperparameters."""
    opt_ml = root / "opt_ml"
    fe = opt_ml / "input" / "data" / "fe"
    fe.mkdir(parents=True)
    if fe_source is not None:
        shutil.copy(fe_source, fe / (fe_name or md5(fe_source)))
    if config is not None:
        (opt_ml / "input" / "data" / "config").mkdir(parents=True)
        shutil.copy(config, opt_ml / "input" / "data" / "config" / config.name)
    if stage is not None:
        (opt_ml / "input" / "config").mkdir(parents=True)
        (opt_ml / "input" / "config" / "hyperparameters.json").write_text(json.dumps({"stage": stage}))
    return opt_ml


def run(opt_ml: Path, *extra: str, env: dict | None = None, cwd: Path = PROJECT):
    """Run the entrypoint as its own Python process (as the train launcher does). MLflow off unless env sets it."""
    environment = {key: value for key, value in os.environ.items() if key != "MLFLOW_TRACKING_URI"}
    environment.update(env or {})
    return subprocess.run([sys.executable, str(ENTRYPOINT), "--opt-ml", str(opt_ml), *extra], cwd=cwd,
                          env=environment, capture_output=True, text=True, check=False)


def failure_text(opt_ml: Path) -> str:
    return (opt_ml / "output" / "failure").read_text()


# ---------------------------------------------------------------------------
# 1. Stage production with the base project's real p5_03 (every SageMaker job)
# ---------------------------------------------------------------------------
def test_production_writes_the_3_n09_files_and_50550(tmp_path):
    need_fe_data()
    laptop_model = PROJECT / "models" / "p5_03_production" / "model_latest.pkl"
    before = (md5(laptop_model) if laptop_model.exists() else None, sorted((PROJECT / "results").glob("*.json")))
    opt_ml = make_opt_ml(tmp_path)
    (opt_ml / "input" / "data" / "fe" / ".DS_Store").write_text("Finder file: skipped")

    result = run(opt_ml)

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    model_dir = opt_ml / "model"
    assert sorted(path.name for path in model_dir.iterdir()) == N09            # nothing else
    metadata = json.loads((model_dir / "metadata_latest.json").read_text())
    evaluation = json.loads((model_dir / "evaluation.json").read_text())
    assert metadata["model_name"] == "p5_03_production"
    assert metadata["metadata"]["best_params"]["C"] == 0.1                    # the base project's fixed YAML, not a grid search
    assert evaluation["business_metrics"]["net_benefit"] == 50550             # the reference number
    assert int(metadata["metrics"]["net_benefit"]) == 50550
    assert md5(opt_ml / "output" / "data" / "p5_02_best_config.yaml") == md5(BASE_CONFIG)
    assert not (opt_ml / "output" / "failure").exists()
    assert "p5_02 skipped" in result.stdout and "=== The base project p5_02" not in result.stdout
    assert "MLFLOW_TRACKING_URI is not set" in result.stdout
    # The base project ran in a work copy: the laptop's own model and results did not change
    after = (md5(laptop_model) if laptop_model.exists() else None, sorted((PROJECT / "results").glob("*.json")))
    assert after == before


def test_stage_defaults_to_production_without_hyperparameters(tmp_path):
    need_fe_data()
    opt_ml = make_opt_ml(tmp_path, stage=None, fe_name="fe1")
    result = run(opt_ml)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "stage production" in result.stdout
    assert sorted(path.name for path in (opt_ml / "model").iterdir()) == N09


def test_mlflow_run_is_logged_when_the_uri_is_set(tmp_path):
    """with MLFLOW_TRACKING_URI set, the job logs one run (here to a SQLite file, no server, no AWS)."""
    need_fe_data()
    from mlflow import MlflowClient

    opt_ml = make_opt_ml(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = run(opt_ml, env={"MLFLOW_TRACKING_URI": uri}, cwd=tmp_path)   # run artifacts go to <tmp>/mlruns
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("ccfd-fraud-detection")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run_data = runs[0].data
    assert run_data.metrics["net_benefit"] == 50550
    assert run_data.tags["journey_stage"] == "production"
    assert run_data.tags["base_script"] == "p5_03_train_production_model.py"
    assert run_data.tags["config_md5"] == md5(BASE_CONFIG)
    assert run_data.tags["data_md5"] == md5(FE_DATA)
    assert run_data.tags["training_job"] == "local" and run_data.tags["runtime"] == "local"
    assert json.loads(result.stdout.split("TRAIN SUMMARY ")[1].splitlines()[0])["mlflow_run_id"] == runs[0].info.run_id
    # The run keeps the job's own evaluation.json (the base project's results JSON, the one shape), byte for byte the same data
    stored = client.download_artifacts(runs[0].info.run_id, "model_dir/evaluation.json", str(tmp_path))
    assert json.loads(Path(stored).read_text()) == json.loads((opt_ml / "model" / "evaluation.json").read_text())


# ---------------------------------------------------------------------------
# 2. Bad inputs: a clear reason in /opt/ml/output/failure and exit code 1 (the base project never starts)
# ---------------------------------------------------------------------------
def test_two_files_in_channel_fe_fail(tmp_path):
    opt_ml = make_opt_ml(tmp_path, fe_source=BASE_CONFIG, fe_name="part-1")
    (opt_ml / "input" / "data" / "fe" / "part-2").write_text("x")
    result = run(opt_ml)
    assert result.returncode == 1
    assert failure_text(opt_ml) == "Channel fe must hold exactly 1 file, found 2 (part-1, part-2)."
    assert not (opt_ml / "model").exists()


def test_empty_channel_fe_fails(tmp_path):
    opt_ml = make_opt_ml(tmp_path, fe_source=None)
    result = run(opt_ml)
    assert result.returncode == 1
    assert "Channel fe must hold exactly 1 file, found 0" in failure_text(opt_ml)


def test_production_without_channel_config_fails(tmp_path):
    opt_ml = make_opt_ml(tmp_path, config=None, fe_source=BASE_CONFIG)
    result = run(opt_ml)
    assert result.returncode == 1
    assert "Stage production needs channel config" in failure_text(opt_ml)


def test_config_without_best_params_fails(tmp_path):
    bad = tmp_path / "p5_02_best_config.yaml"
    bad.write_text("method: something else\n")
    opt_ml = make_opt_ml(tmp_path, config=bad, fe_source=BASE_CONFIG)
    result = run(opt_ml)
    assert result.returncode == 1
    assert "has no best_params" in failure_text(opt_ml)


def test_unknown_stage_fails(tmp_path):
    opt_ml = make_opt_ml(tmp_path, stage="tuning", fe_source=BASE_CONFIG)
    result = run(opt_ml)
    assert result.returncode == 1
    assert failure_text(opt_ml) == "Unknown stage 'tuning'. Use one of: production, gridsearch."


# ---------------------------------------------------------------------------
# 3. Stage gridsearch with stand-in scripts (the order of the base project's scripts and the files they leave)
# ---------------------------------------------------------------------------
STAND_IN_P5_02 = """
import os
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/p5_02_best_config.yaml", "w") as f:
    f.write("best_params:\\n  C: 10\\nbest_cv_score: 34960.0\\ntest_net_benefit: 47925\\n")
with open("order.log", "a") as f:
    f.write("p5_02\\n")
"""
STAND_IN_P5_03 = """
import json, os, sys
import yaml
config = yaml.safe_load(open("artifacts/p5_02_best_config.yaml"))
assert os.path.exists("data/credit_card_transactions_fe1_amount_ratio.csv")
os.makedirs("models/p5_03_production", exist_ok=True)
os.makedirs("results", exist_ok=True)
open("models/p5_03_production/model_latest.pkl", "wb").write(b"stand-in")
json.dump({"model_name": "p5_03_production", "metrics": {"net_benefit": "47925"},
           "metadata": {"best_params": config["best_params"]}}, open("models/p5_03_production/metadata_latest.json", "w"))
json.dump({"business_metrics": {"net_benefit": 47925}},
          open("results/logistic_regression_p5_03_production_20261012_120000.json", "w"))
with open("order.log", "a") as f:
    f.write("p5_03\\n")
print(open("order.log").read().replace(chr(10), " "))
sys.exit(int(os.environ.get("STAND_IN_EXIT", "0")))
"""


@pytest.fixture
def stand_in_code(tmp_path):
    """A code folder with the base project's file names, holding the 2 stand-in scripts and an empty utils/."""
    code = tmp_path / "code"
    (code / "utils").mkdir(parents=True)
    (code / "utils" / "__init__.py").write_text("")
    (code / "p5_02_class_weight_gridsearch.py").write_text(STAND_IN_P5_02)
    (code / "p5_03_train_production_model.py").write_text(STAND_IN_P5_03)
    return code


def test_gridsearch_runs_p5_02_then_p5_03_and_keeps_its_yaml(tmp_path, stand_in_code):
    opt_ml = make_opt_ml(tmp_path, stage="gridsearch", fe_source=BASE_CONFIG)   # config channel present: not used
    result = run(opt_ml, "--code-dir", str(stand_in_code))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "p5_02 p5_03" in result.stdout                                       # the order, from the work copy
    assert "channel config is not used" in result.stdout
    assert sorted(path.name for path in (opt_ml / "model").iterdir()) == N09
    assert "C: 10" in (opt_ml / "output" / "data" / "p5_02_best_config.yaml").read_text()
    summary = json.loads(result.stdout.split("TRAIN SUMMARY ")[1].splitlines()[0])
    assert (summary["stage"], summary["best_params"], summary["net_benefit"]) == ("gridsearch", {"C": 10}, 47925)
    assert list(summary["seconds"]) == ["p5_02_class_weight_gridsearch.py", "p5_03_train_production_model.py"]
    assert not (stand_in_code / "order.log").exists()                           # the code folder is never written


def test_a_failing_base_script_is_the_failure_reason(tmp_path, stand_in_code):
    opt_ml = make_opt_ml(tmp_path, fe_source=BASE_CONFIG)
    result = run(opt_ml, "--code-dir", str(stand_in_code), env={"STAND_IN_EXIT": "3"})
    assert result.returncode == 1
    assert failure_text(opt_ml) == ("The base project's p5_03_train_production_model.py failed (exit code 3); "
                                    "its own output is in the log above.")


def test_mlflow_is_not_imported_when_the_uri_is_not_set(tmp_path, stand_in_code):
    opt_ml = make_opt_ml(tmp_path, fe_source=BASE_CONFIG)
    code = ("import sys; sys.argv = ['m08_03']; import m08_03_train_entrypoint as entry; "
            f"code = entry.main(['--opt-ml', {str(opt_ml)!r}, '--code-dir', {str(stand_in_code)!r}]); "
            "print('EXIT', code, 'MLFLOW_IMPORTED', 'mlflow' in sys.modules)")
    environment = {key: value for key, value in os.environ.items() if key != "MLFLOW_TRACKING_URI"}
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=environment, capture_output=True,
                            text=True, check=True)
    assert "EXIT 0 MLFLOW_IMPORTED False" in result.stdout


# ---------------------------------------------------------------------------
# 4. The image recipe and the train launcher
# ---------------------------------------------------------------------------
def dockerfile_lines(name: str) -> list[str]:
    return [line.strip() for line in (PROJECT / name).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_dockerfile_training_recipe():
    lines = dockerfile_lines("Dockerfile.training")
    serving_base = next(line for line in dockerfile_lines("Dockerfile.serving") if line.startswith("FROM "))
    assert lines[0] == serving_base and "@sha256:" in serving_base            # the same base, by digest
    assert not [line for line in lines if line.upper().startswith("USER ")]  # no USER line
    assert not [line for line in lines if line.upper().startswith("ENTRYPOINT")]
    assert ("RUN pip install --only-binary=:all: -r requirements-training.txt -c constraints-training.txt"
            in lines)  # installed with the constraints file
    assert "COPY m08_03_train_entrypoint.py ./" in lines
    assert "COPY m08_02_data_checks.py m13_03_evaluate_vs_champion.py m15_03_drift_job.py ./" in lines   # processing entrypoints
    text = (PROJECT / "Dockerfile.training").read_text()
    assert "p5_03_train_production_model.py \\\n     p5_04_inference_api.py ./" in text   # the base project's p5_04 for m08_02
    assert "COPY --chmod=0755 train ./" in lines
    assert lines[-1] == 'CMD ["train"]'
    assert "the newest built-in scikit-learn image is 1.9.0 on Python 3.12" in text   # the why-our-own-container sentence
    assert "scikit-learn 1.8.0 on Python 3.14" in text


ENTRYPOINTS = ("m08_02_data_checks.py", "m08_03_train_entrypoint.py", "m13_03_evaluate_vs_champion.py")


def copied_into_training_image() -> set[str]:
    """Every source of every COPY line in Dockerfile.training (folders without their slash)."""
    text = (PROJECT / "Dockerfile.training").read_text().replace("\\\n", " ")
    sources = set()
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["COPY"]:
            sources |= {part.rstrip("/") for part in parts[1:-1] if not part.startswith("--")}
    return sources


def project_modules(path: Path, found: set[str]) -> set[str]:
    """The project's own top-level modules and packages a file imports (also inside functions), followed further."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for top in {name.split(".")[0] for name in names} - found:
            if (PROJECT / f"{top}.py").is_file():
                found.add(top)
                project_modules(PROJECT / f"{top}.py", found)
            elif (PROJECT / top / "__init__.py").is_file():
                found.add(top)
                for module in sorted((PROJECT / top).glob("*.py")):
                    project_modules(module, found)
    return found


def test_every_entrypoint_import_is_copied_into_the_training_image():
    """A missing COPY (as p5_04 was before Step-08) shows up here, not first in a container run."""
    import m08_02_data_checks
    import m08_03_train_entrypoint

    copied = copied_into_training_image()
    needed = set()
    for entrypoint in ENTRYPOINTS:
        needed.add(entrypoint)
        needed |= {name if (PROJECT / name).is_dir() else f"{name}.py"
                   for name in project_modules(PROJECT / entrypoint, set())}
    # the base project scripts the entrypoints run as "python <script>"
    needed |= {m08_02_data_checks.P4_01_SCRIPT, m08_03_train_entrypoint.GRIDSEARCH_SCRIPT,
               m08_03_train_entrypoint.PRODUCTION_SCRIPT, "train"}
    assert {"mlops_utils", "utils", "p5_04_inference_api.py"} <= needed
    assert needed - copied == set(), f"not copied into Dockerfile.training: {sorted(needed - copied)}"


def test_the_training_image_imports_its_entrypoints_and_runs_train(docker_image):
    """With MLOPS_TRAINING_IMAGE set: the built image imports the 3 entrypoints and runs the train launcher."""
    image = docker_image("MLOPS_TRAINING_IMAGE", "ccfd-training:dev")
    name = f"ccfd-t1-training-image-{os.getpid()}"
    script = ("python -c 'import m08_02_data_checks, m08_03_train_entrypoint, m13_03_evaluate_vs_champion' "
              "&& train --help > /dev/null && echo IMAGE-OK")
    try:
        result = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--name", name,
                                 "--entrypoint", "sh", image, "-c", script],
                                capture_output=True, text=True, check=False, timeout=300)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    assert result.returncode == 0 and "IMAGE-OK" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]


def test_train_launcher_execs_the_entrypoint():
    """The launcher file itself (it runs for real in the image test above and in Step-07 T2)."""
    launcher = PROJECT / "train"
    assert os.access(launcher, os.X_OK)
    text = launcher.read_bytes()
    assert b"\r" not in text and text.startswith(b"#!/bin/sh\n")
    assert b'exec python "$APP_DIR/m08_03_train_entrypoint.py" "$@"' in text
    assert "train text eol=lf" in (PROJECT / ".gitattributes").read_text()
