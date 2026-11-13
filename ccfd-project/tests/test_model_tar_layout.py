"""
The model.tar.gz layout.

Every course model reaches SageMaker as one model.tar.gz with exactly 3 files at its root:
model_latest.pkl, metadata_latest.json and evaluation.json. SageMaker unpacks it into /opt/ml/model, where the
serving image reads it. These tests make each kind of tar, unpack it and start the serving API on it (/ping 200):
  - the laptop packages of tests/package_model_tar.py: the base project's p3_01, p5_01 (the weaker model) and p5_03
  - a training output: m08_03_train_entrypoint.py's /opt/ml/model folder, packed the way SageMaker packs it
  - sm-local/model, the output of the local Linux container run (Step-07 T2), when it is there

The serving API runs inside this Python process (the same code as the image). To start the real serving image on
each unpacked folder as well, set MLOPS_SERVING_IMAGE (Docker must run; the folders are unpacked under sm-local/,
inside the ccfd-project folder, so Docker on the laptop can mount them):
    MLOPS_SERVING_IMAGE=ccfd-serving:dev python -m pytest -q tests/test_model_tar_layout.py
Without the variable, test_the_serving_image_check_runs is skipped, so the summary shows that the image was not
started (MLOPS_REQUIRE_IMAGE=1 makes it a failure; see tests/conftest.py).

The laptop models come from the base project's chain (Step-01); without them the tests skip, unless MLOPS_REQUIRE_MODEL=1.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import package_model_tar as packager  # tests/package_model_tar.py (pytest puts tests/ on sys.path)
import pytest
from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
N09 = ["model_latest.pkl", "metadata_latest.json", "evaluation.json"]
# The base project's Improvement Journey on the base project's own test split
LAPTOP_MODELS = {"p3_01_baseline": 39625, "p5_01_randomsearch": 37025, "p5_03_production": 50550}
PAYLOAD = PROJECT / "tests" / "payloads" / "sample_transactions.json"
UNPACK_ROOT = PROJECT / "sm-local" / "tar-check"


def need(path: Path, what: str):
    if not path.exists():
        message = f"needs {what}: run the base project's chain (p1_01 ... p5_03) first"
        if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
            pytest.fail(f"MLOPS_REQUIRE_MODEL=1, but this test {message}")
        pytest.skip(message)


def sagemaker_tar(folder: Path, tar_path: Path) -> Path:
    """Pack a /opt/ml/model folder the way a SageMaker training job does: every file at the tar root."""
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(folder.iterdir()):
            tar.add(path, arcname=path.name)
    return tar_path


def check_layout_and_unpack(tar_path: Path, dest: Path) -> Path:
    """The tar holds exactly the 3 model files, as plain files at the root; unpack it into dest."""
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        assert sorted(member.name for member in members) == sorted(N09)
        assert all(member.isfile() for member in members)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        tar.extractall(dest, filter="data")
    return dest


def ping_in_process(model_dir: Path, transactions: list) -> tuple[int, list]:
    """Start the serving API on model_dir (as the image does with MODEL_DIR=/opt/ml/model); /ping and one call."""
    import m01_03_serving_api as api

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "MODEL_DIR", model_dir)
        with TestClient(api.app) as client:
            return client.get("/ping").status_code, client.post("/invocations", json=transactions).json()


def ping_serving_image(image: str, model_dir: Path, timeout: float = 90) -> int:
    """docker run <image> serve with model_dir mounted read only at /opt/ml/model; the /ping status code."""
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "--platform", "linux/amd64", "-p", "127.0.0.1::8080",
         "-v", f"{model_dir}:/opt/ml/model:ro", image, "serve"],
        capture_output=True, text=True, check=True)
    container = started.stdout.strip()
    try:
        port = subprocess.run(["docker", "port", container, "8080/tcp"], capture_output=True, text=True,
                              check=True).stdout.split(":")[-1].strip()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=3) as response:
                    return response.status
            except OSError:
                time.sleep(1)
        return 0
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


@pytest.fixture(scope="module")
def transactions():
    return json.loads(PAYLOAD.read_text())


def check_tar(tar_path: Path, case: str, expected_name: str, expected_net_benefit: int, transactions: list):
    image = os.environ.get("MLOPS_SERVING_IMAGE")
    # Docker on the laptop mounts only folders under the home folder: unpack under sm-local/ when the image runs
    model_dir = check_layout_and_unpack(tar_path, (UNPACK_ROOT if image else tar_path.parent) / case)
    try:
        metadata = json.loads((model_dir / "metadata_latest.json").read_text())
        evaluation = json.loads((model_dir / "evaluation.json").read_text())
        assert metadata["model_name"] == expected_name
        assert evaluation["business_metrics"]["net_benefit"] == expected_net_benefit
        status, rows = ping_in_process(model_dir, transactions)
        assert status == 200
        assert {row["model_name"] for row in rows} == {expected_name}       # the API serves THIS model
        if image:
            assert ping_serving_image(image, model_dir) == 200
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
        if image and UNPACK_ROOT.is_dir() and not any(UNPACK_ROOT.iterdir()):
            UNPACK_ROOT.rmdir()


# ---------------------------------------------------------------------------
# 1. The laptop packages (m09_02): p3_01, p5_01, p5_03
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("folder", sorted(LAPTOP_MODELS))
def test_packaged_laptop_model_has_the_n09_layout_and_serves(folder, tmp_path, transactions):
    need(MODELS / folder / "model_latest.pkl", f"the laptop model models/{folder}")
    tar_path = tmp_path / f"{folder}.tar.gz"
    result = packager.package(MODELS / folder, str(tar_path))
    assert result["files"] == N09 and result["net_benefit"] == LAPTOP_MODELS[folder]
    check_tar(tar_path, f"packaged-{folder}", folder, LAPTOP_MODELS[folder], transactions)


def test_package_is_the_same_bytes_every_time(tmp_path):
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    packager.package(MODELS / "p5_01_randomsearch", str(first))
    time.sleep(1.1)
    packager.package(MODELS / "p5_01_randomsearch", str(second))
    assert first.read_bytes() == second.read_bytes()


def test_cli_default_is_the_weaker_p5_01(tmp_path):
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    output = tmp_path / "weaker.tar.gz"
    result = subprocess.run([sys.executable, "tests/package_model_tar.py", "--output", str(output)],
                            cwd=PROJECT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Packaged p5_01_randomsearch (Net Benefit 37,025 USD)" in result.stdout
    with tarfile.open(output) as tar:
        assert tar.getnames() == N09


def test_a_results_file_of_another_model_is_refused(tmp_path):
    need(MODELS / "p5_03_production" / "model_latest.pkl", "the laptop model models/p5_03_production")
    other = sorted((PROJECT / "results").glob("*p3_01_baseline*.json"))
    need(other[-1] if other else PROJECT / "results" / "missing", "The base project's p3_01 results JSON")
    with pytest.raises(ValueError, match="belongs to another model"):
        packager.package(MODELS / "p5_03_production", str(tmp_path / "x.tar.gz"), evaluation=other[-1])


def test_a_folder_without_results_is_refused(tmp_path):
    folder = tmp_path / "p9_99_unknown"
    folder.mkdir()
    (folder / "model_latest.pkl").write_bytes(b"x")
    (folder / "metadata_latest.json").write_text(json.dumps({"model_name": "p9_99_unknown"}))
    with pytest.raises(FileNotFoundError, match="No base project results file for p9_99_unknown"):
        packager.package(folder, str(tmp_path / "x.tar.gz"), results_dir=tmp_path)


class FakeS3:
    """Records upload_file calls (the only S3 call m09_02 makes) and keeps a copy of the uploaded bytes."""

    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((bucket, key, Path(filename).read_bytes()))


def test_s3_output_uploads_the_same_tar(tmp_path):
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    s3 = FakeS3()
    uri = "s3://ccfd-mlops-jobs-a1b2c3/models/p5_01_randomsearch/model.tar.gz"     # the key of MLOps 09
    packager.package(MODELS / "p5_01_randomsearch", uri, s3_client=s3)
    packager.package(MODELS / "p5_01_randomsearch", str(tmp_path / "local.tar.gz"))
    assert [(bucket, key) for bucket, key, _ in s3.uploads] == [
        ("ccfd-mlops-jobs-a1b2c3", "models/p5_01_randomsearch/model.tar.gz")]
    assert s3.uploads[0][2] == (tmp_path / "local.tar.gz").read_bytes()


@pytest.mark.parametrize("uri", ["s3://ccfd-mlops-jobs-a1b2c3", "s3://ccfd-mlops-jobs-a1b2c3/models/"])
def test_s3_uri_must_name_a_file(uri):
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    with pytest.raises(ValueError, match="not an s3://bucket/key URI of a file"):
        packager.package(MODELS / "p5_01_randomsearch", uri, s3_client=FakeS3())


def test_an_s3_error_is_printed_with_the_account_id_masked(tmp_path, monkeypatch, capsys):
    """An S3 AccessDenied names the caller's ARN: main() prints it masked and exits 1."""
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    import boto3
    from botocore.exceptions import ClientError

    account = "".join(str(digit % 10) for digit in range(1, 13))      # a made-up 12-digit number
    denied = {"Error": {"Code": "AccessDenied", "Message": f"User: arn:aws:iam::{account}:user/mlops-test-user is not "
                                                           "authorized to perform: s3:PutObject"}}

    class DeniedS3:
        def upload_file(self, filename, bucket, key):
            raise ClientError(denied, "PutObject")

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: DeniedS3())
    with pytest.raises(SystemExit) as stop:
        packager.main(["--output", "s3://ccfd-mlops-jobs-a1b2c3/models/p5_01_randomsearch/model.tar.gz"])
    out = capsys.readouterr()
    assert stop.value.code == 1
    assert "ERROR: ClientError" in out.err and "arn:aws:iam::<ACCOUNT_ID>:user/mlops-test-user" in out.err
    assert account not in out.out + out.err


def test_an_evaluation_json_in_another_shape_is_refused(tmp_path):
    """evaluation.json is the base project's results JSON; a {"metrics", "lineage"} file is never packed."""
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    other = tmp_path / "evaluation.json"
    other.write_text(json.dumps({"metrics": {"net_benefit": 37025}, "lineage": {}}))
    with pytest.raises(ValueError, match="is not the base project's results JSON"):
        packager.package(MODELS / "p5_01_randomsearch", str(tmp_path / "x.tar.gz"), evaluation=other)


def test_local_output_does_not_import_boto3(tmp_path):
    need(MODELS / "p5_01_randomsearch" / "model_latest.pkl", "the laptop model models/p5_01_randomsearch")
    code = ("import sys; sys.path.insert(0, 'tests'); import package_model_tar as p; "
            f"p.package(p.DEFAULT_MODEL_DIR, {str(tmp_path / 'x.tar.gz')!r}); print('boto3' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# 2. Training outputs: the entrypoint's /opt/ml/model, packed as SageMaker packs it
# ---------------------------------------------------------------------------
def test_training_output_tar_has_the_n09_layout_and_serves(tmp_path, transactions):
    fe_data = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
    need(fe_data, "the laptop FE1 data")
    opt_ml = tmp_path / "opt_ml"
    (opt_ml / "input" / "data" / "fe").mkdir(parents=True)
    (opt_ml / "input" / "data" / "config").mkdir(parents=True)
    shutil.copy(fe_data, opt_ml / "input" / "data" / "fe" / "fe1")
    shutil.copy(PROJECT / "config" / "p5_02_best_config.yaml", opt_ml / "input" / "data" / "config")
    environment = {key: value for key, value in os.environ.items() if key != "MLFLOW_TRACKING_URI"}
    result = subprocess.run([sys.executable, "m08_03_train_entrypoint.py", "--opt-ml", str(opt_ml)], cwd=PROJECT,
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    tar_path = sagemaker_tar(opt_ml / "model", tmp_path / "model.tar.gz")
    check_tar(tar_path, "training-output", "p5_03_production", 50550, transactions)


def test_local_container_output_has_the_n09_layout_and_serves(tmp_path, transactions):
    """sm-local/model: the production output of the local Linux container run (Step-07 T2); skipped without it."""
    folder = PROJECT / "sm-local" / "model"
    if not (folder / "model_latest.pkl").exists():
        pytest.skip("no sm-local/model yet: run the training image locally (Step-07 T2)")
    tar_path = sagemaker_tar(folder, tmp_path / "model.tar.gz")
    check_tar(tar_path, "local-container-output", "p5_03_production", 50550, transactions)


def test_the_serving_image_check_runs(docker_image):
    """
    The tar tests above start the real serving image only when MLOPS_SERVING_IMAGE is set. This test makes that
    visible: without the variable it is skipped (or fails under MLOPS_REQUIRE_IMAGE=1); with it, the image must be
    there, built for linux/amd64.
    """
    image = docker_image("MLOPS_SERVING_IMAGE", "ccfd-serving:dev")
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Os}}/{{.Architecture}}"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "linux/amd64"


def test_tar_with_an_extra_file_is_caught(tmp_path):
    """The layout check itself: a 4th file (or a sub folder) at the root is not the model.tar.gz layout."""
    folder = tmp_path / "model"
    folder.mkdir()
    for name in [*N09, "notes.txt"]:
        (folder / name).write_bytes(b"x")
    with pytest.raises(AssertionError):
        check_layout_and_unpack(sagemaker_tar(folder, tmp_path / "m.tar.gz"), tmp_path / "out")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for name in N09:
            info = tarfile.TarInfo(f"model/{name}")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    (tmp_path / "nested.tar.gz").write_bytes(raw.getvalue())
    with pytest.raises(AssertionError):
        check_layout_and_unpack(tmp_path / "nested.tar.gz", tmp_path / "out2")
