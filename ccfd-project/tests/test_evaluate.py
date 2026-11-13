"""
Tests for m13_03_evaluate_vs_champion.py, the gate step of the pipeline.

The models are the base project's laptop models (Step-01), packed into model.tar.gz files with tests/package_model_tar.py
(the model.tar.gz layout). The gate data is the fixed gate holdout of Step-05, data/gate_holdout.csv (2,000 rows,
60 frauds). On it, at threshold 0.5:
    p5_03 36,975 (x, recorded in manifest.yaml as gate_x)     p5_01 22,600     p3_01 19,225
So p5_03 vs p3_01 is PROMOTE, p5_01 vs p5_03 is REJECT, and p5_03 vs p5_03 is a tie at x, which is REJECT.

Each test builds the processing job's folders (the candidate tar in input/model, the gate file in input/gate), as the
pipeline does. boto3 never reaches AWS here: the registry calls go to botocore's Stubber, which checks the exact
request, and the champion's S3 download goes to a small fake client that copies a local tar. The made-up account ID is
built at run time.
The tests skip without the laptop models or the gate holdout, unless MLOPS_REQUIRE_MODEL=1 (see tests/conftest.py).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import boto3
import package_model_tar as packager  # tests/package_model_tar.py (pytest puts tests/ on sys.path)
import pytest
from botocore.stub import Stubber

import m13_03_evaluate_vs_champion as m13

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
GATE = PROJECT / "data" / "gate_holdout.csv"
GATE_X = 36975                                          # p5_03 on the gate holdout (manifest.yaml gate_x)
NET_BENEFIT = {"p3_01_baseline": 19225, "p5_01_randomsearch": 22600, "p5_03_production": GATE_X}
ACCOUNT = "".join(str(digit % 10) for digit in range(1, 13))       # a made-up 12-digit number, built at run time
GROUP_ARN = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/ccfd-fraud-detector"
BUCKET = "ccfd-mlops-jobs-a1b2c3"
LOOKUP = {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelApprovalStatus": "Approved",
          "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 10}
GROUP = {"ModelPackageGroupName": "ccfd-fraud-detector"}


def need(path: Path, what: str):
    if not path.exists():
        message = f"needs {what}: run the base project's chain (Step-01) and m03_03's gate holdout command (Step-05) first"
        if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
            pytest.fail(f"MLOPS_REQUIRE_MODEL=1, but this test {message}")
        pytest.skip(message)


@pytest.fixture(scope="module")
def tars(tmp_path_factory):
    """model.tar.gz of p3_01, p5_01 and p5_03, made by m09_02 (the model.tar.gz layout)."""
    need(GATE, "the gate holdout data/gate_holdout.csv")
    folder = tmp_path_factory.mktemp("packages")
    made = {}
    for name in NET_BENEFIT:
        need(MODELS / name / "model_latest.pkl", f"the laptop model models/{name}")
        made[name] = folder / f"{name}.tar.gz"
        packager.package(MODELS / name, str(made[name]))
    return made


def layout(root: Path, candidate: Path, gate_data: Path = GATE, gate_name: str = "gate_holdout.csv") -> Path:
    """The processing job's input folders: input/model/model.tar.gz and input/gate/<one file>."""
    (root / "input" / "model").mkdir(parents=True)
    (root / "input" / "gate").mkdir(parents=True)
    shutil.copy(candidate, root / "input" / "model" / "model.tar.gz")
    shutil.copy(gate_data, root / "input" / "gate" / gate_name)
    return root


def gate_run(root: Path, candidate: Path, sm, s3=None, **layout_options) -> dict:
    return m13.evaluate(layout(root, candidate, **layout_options), sm=sm, s3=s3)


def decision_file(root: Path) -> dict:
    return json.loads((root / "output" / "gate" / "gate_decision.json").read_text())


# ---------------------------------------------------------------------------
# 1. The champion from the SageMaker registry (stubbed boto3)
# ---------------------------------------------------------------------------
def test_gate_x_is_the_manifest_value():
    for path in (PROJECT / "manifest.yaml", PROJECT.parent / "manifest.yaml"):
        if path.is_file():
            import yaml

            assert yaml.safe_load(path.read_text())["gate_x"]["value"] == GATE_X
            return
    pytest.skip("manifest.yaml not found next to ccfd-project/ or in it")


class FakeS3:
    """download_file (the only S3 call m13_03 makes): copies the local tar that stands for the S3 object."""

    def __init__(self, objects: dict):
        self.objects, self.downloads = objects, []

    def download_file(self, bucket, key, filename):
        self.downloads.append(f"s3://{bucket}/{key}")
        shutil.copy(self.objects[f"s3://{bucket}/{key}"], filename)


@pytest.fixture
def sm():
    """A SageMaker client with made-up keys (nothing is ever sent) and a list of the API calls it makes."""
    client = boto3.client("sagemaker", region_name="us-east-1",
                          aws_access_key_id="testing", aws_secret_access_key="testing")
    client.calls = []
    client.meta.events.register("provide-client-params.sagemaker.*",
                                lambda params, model, **_: client.calls.append(model.name))
    with Stubber(client) as stubber:
        client.stubber = stubber
        yield client
        stubber.assert_no_pending_responses()


def package_arn(version: int) -> str:
    return f"{GROUP_ARN}/{version}"


def described(version: int, url: str | None) -> dict:
    """A describe_model_package answer for one version of ccfd-fraud-detector."""
    container = {"Image": f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/ccfd-serving:sha-abc1234"}
    if url:
        container["ModelDataUrl"] = url
    return {
        "ModelPackageName": "ccfd-fraud-detector", "ModelPackageGroupName": "ccfd-fraud-detector",
        "ModelPackageVersion": version, "ModelPackageArn": package_arn(version),
        "CreationTime": datetime(2026, 10, 12, tzinfo=UTC), "ModelPackageStatus": "Completed",
        "ModelPackageStatusDetails": {"ValidationStatuses": []}, "ModelApprovalStatus": "Approved",
        "InferenceSpecification": {"Containers": [container], "SupportedContentTypes": ["application/json"],
                                   "SupportedResponseMIMETypes": ["application/json"]},
    }


def group_exists(sm) -> None:
    """The describe_model_package_group answer m13_03 reads before the lookup (the group must exist)."""
    sm.stubber.add_response("describe_model_package_group", {
        "ModelPackageGroupName": "ccfd-fraud-detector", "ModelPackageGroupArn": GROUP_ARN.replace(
            ":model-package/", ":model-package-group/"),
        "CreationTime": datetime(2026, 10, 12, tzinfo=UTC), "ModelPackageGroupStatus": "Completed",
        "CreatedBy": {}}, GROUP)


def listed(*versions: int) -> dict:
    return {"ModelPackageSummaryList": [
        {"ModelPackageName": "ccfd-fraud-detector", "ModelPackageGroupName": "ccfd-fraud-detector",
         "ModelPackageVersion": version, "ModelPackageArn": package_arn(version),
         "CreationTime": datetime(2026, 10, 12, tzinfo=UTC), "ModelPackageStatus": "Completed",
         "ModelApprovalStatus": "Approved"} for version in versions]}


def champion_in_registry(sm, tar: Path, version: int = 3) -> FakeS3:
    """The newest Approved package is `version`; its ModelDataUrl is a model.tar.gz that the fake S3 serves from tar."""
    url = f"s3://{BUCKET}/models/v{version}/model.tar.gz"
    group_exists(sm)
    sm.stubber.add_response("list_model_packages", listed(version), LOOKUP)
    sm.stubber.add_response("describe_model_package", described(version, url), {"ModelPackageName": package_arn(version)})
    return FakeS3({url: tar})


# ---------------------------------------------------------------------------
# 2. The 3 decisions on the gate holdout
# ---------------------------------------------------------------------------
def test_promote_p5_03_over_p3_01(tmp_path, tars, sm):
    s3 = champion_in_registry(sm, tars["p3_01_baseline"])
    record = gate_run(tmp_path, tars["p5_03_production"], sm, s3)
    assert (record["decision"], record["promote"]) == ("PROMOTE", 1)
    assert record["candidate"]["net_benefit"] == GATE_X and record["champion"]["net_benefit"] == 19225
    assert record["net_benefit_gain"] == GATE_X - 19225 == 17750
    assert record["candidate"]["model_name"] == "p5_03_production"
    assert record["candidate"]["source"] == "file model.tar.gz"
    assert record["champion"]["source"] == "package ccfd-fraud-detector/3 (Approved)"
    assert record["champion_choice"] == "the newest Approved package in ccfd-fraud-detector"
    assert (record["gate_data"], record["gate_rows"], record["gate_frauds"]) == ("gate_holdout.csv", 2000, 60)
    assert len(record["gate_data_md5"]) == 32
    assert s3.downloads == [f"s3://{BUCKET}/models/v3/model.tar.gz"]
    assert sm.calls == ["DescribeModelPackageGroup", "ListModelPackages", "DescribeModelPackage"]
    assert decision_file(tmp_path) == record                     # the file the pipeline's condition reads
    assert decision_file(tmp_path)["promote"] == 1
    text = (tmp_path / "output" / "gate" / "gate_decision.json").read_text()
    assert not re.search(r"(?<!\d)\d{12}(?!\d)", text)                      # no account ID in the file


def test_reject_p5_01_against_p5_03(tmp_path, tars, sm):
    s3 = champion_in_registry(sm, tars["p5_03_production"])
    record = gate_run(tmp_path, tars["p5_01_randomsearch"], sm, s3)
    assert (record["decision"], record["promote"]) == ("REJECT", 0)
    assert (record["candidate"]["net_benefit"], record["champion"]["net_benefit"]) == (22600, GATE_X)
    assert record["net_benefit_gain"] == -14375
    assert record["serve_model_name"] == "p5_03_production"           # the champion stays


def test_tie_is_reject_at_x(tmp_path, tars, sm):
    s3 = champion_in_registry(sm, tars["p5_03_production"], version=5)
    record = gate_run(tmp_path, tars["p5_03_production"], sm, s3)
    assert record["candidate"]["net_benefit"] == record["champion"]["net_benefit"] == GATE_X
    assert (record["decision"], record["promote"], record["net_benefit_gain"]) == ("REJECT", 0, 0)
    assert "a tie keeps the champion" in record["reason"]


def test_no_champion_yet_promotes_the_candidate(tmp_path, tars, sm):
    group_exists(sm)
    sm.stubber.add_response("list_model_packages", listed(), LOOKUP)
    record = gate_run(tmp_path, tars["p5_01_randomsearch"], sm)
    assert (record["decision"], record["promote"]) == ("PROMOTE", 1)
    assert record["champion"] is None and record["net_benefit_gain"] is None
    assert record["candidate"]["net_benefit"] == 22600
    assert record["champion_choice"].startswith("none: no Approved package in ccfd-fraud-detector yet")
    assert sm.calls == ["DescribeModelPackageGroup", "ListModelPackages"]  # the group, one lookup, nothing else
    assert decision_file(tmp_path)["promote"] == 1


def test_a_missing_package_group_fails_and_is_never_no_champion(tmp_path, tars, sm):
    """An empty list for a missing group would PROMOTE: m13_03 reads the group first and stops."""
    sm.stubber.add_client_error("describe_model_package_group", "ValidationException",
                                f"ModelPackageGroup arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package-group/"
                                "ccfd-fraud-detector does not exist.", 400, expected_params=GROUP)
    with pytest.raises(m13.EvaluateError, match="cannot be read .ValidationException.*never 'no champion yet'") \
            as caught:
        gate_run(tmp_path, tars["p5_03_production"], sm)
    assert ACCOUNT not in str(caught.value)
    # The hint names the student path only (MLOps 09, or the MLOps 16 page after a full exit)
    assert "MLOps 09 creates it" in str(caught.value) and "Coming Back After a Full Exit" in str(caught.value)
    assert sm.calls == ["DescribeModelPackageGroup"]                       # no lookup, no decision
    assert not (tmp_path / "output" / "gate" / "gate_decision.json").exists()


def test_package_without_model_data_url_fails(tmp_path, tars, sm):
    group_exists(sm)
    sm.stubber.add_response("list_model_packages", listed(2), LOOKUP)
    sm.stubber.add_response("describe_model_package", described(2, None), {"ModelPackageName": package_arn(2)})
    with pytest.raises(m13.EvaluateError, match="ccfd-fraud-detector/2 .* has no .*ModelDataUrl"):
        gate_run(tmp_path, tars["p5_03_production"], sm)


def test_an_aws_error_fails_the_job_with_the_account_id_masked(tmp_path, tars, sm, monkeypatch, capsys):
    group_exists(sm)
    sm.stubber.add_client_error("list_model_packages", "AccessDeniedException",
                                f"User: arn:aws:sts::{ACCOUNT}:assumed-role/x is not authorized", 400,
                                expected_params=LOOKUP)
    monkeypatch.setattr(m13, "sagemaker_client", lambda sm_=None: sm)
    layout(tmp_path, tars["p5_03_production"])
    with pytest.raises(SystemExit) as stop:
        m13.main(["--processing-dir", str(tmp_path)])
    assert stop.value.code == 1
    out = capsys.readouterr()
    assert "FAILED: ClientError" in out.out and "<ACCOUNT_ID>" in out.out + out.err
    assert ACCOUNT not in out.out + out.err
    assert not (tmp_path / "output" / "gate" / "gate_decision.json").exists()


# ---------------------------------------------------------------------------
# 3. Inputs: any gate file name, bad tars, bad gate data, the command line
# ---------------------------------------------------------------------------
def test_the_gate_file_may_have_any_name(tmp_path, tars, sm):
    """MLOps 16 passes month_4 as GateHoldoutS3Uri: the job reads the one file in input/gate, whatever its name."""
    s3 = champion_in_registry(sm, tars["p3_01_baseline"])
    record = gate_run(tmp_path, tars["p5_01_randomsearch"], sm, s3, gate_name="month_4_holdout.csv")
    assert (record["decision"], record["net_benefit_gain"]) == ("PROMOTE", 22600 - 19225)
    assert record["gate_data"] == "month_4_holdout.csv"


def test_candidate_not_in_the_n09_layout_fails(tmp_path, tars, sm):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(MODELS / "p5_03_production" / "model_latest.pkl", arcname="model_latest.pkl")
    with pytest.raises(m13.EvaluateError, match="not in the model.tar.gz layout: metadata_latest.json, evaluation.json"):
        gate_run(tmp_path / "job", bad, sm)
    assert sm.calls == []                                                  # stops before the registry


@pytest.mark.parametrize("cut", ["half", "last-20-bytes"])
def test_a_truncated_tar_is_a_clean_failure(tmp_path, tars, sm, cut):
    """A partial download: the gzip stream ends too early. A clear EvaluateError, not a traceback."""
    data = tars["p5_03_production"].read_bytes()
    short = tmp_path / "short.tar.gz"
    short.write_bytes(data[: len(data) // 2] if cut == "half" else data[:-20])
    with pytest.raises(m13.EvaluateError, match="model.tar.gz is not a complete model.tar.gz"):
        gate_run(tmp_path / "job", short, sm)


@pytest.mark.parametrize("problem", ["no-label", "header-only"])
def test_bad_gate_data_is_a_clean_failure(tmp_path, tars, sm, problem):
    import pandas as pd

    data = pd.read_csv(GATE, nrows=50)
    bad = tmp_path / "bad.csv"
    if problem == "no-label":
        data.drop(columns=["is_fraud"]).to_csv(bad, index=False)
        expected = "gate data: gate.csv has no is_fraud column"
    else:
        data.head(0).to_csv(bad, index=False)
        expected = "gate data: gate.csv has a header but no rows"
    with pytest.raises(m13.EvaluateError, match=expected):
        gate_run(tmp_path / "job", tars["p5_03_production"], sm, gate_data=bad, gate_name="gate.csv")
    assert not (tmp_path / "job" / "output" / "gate" / "gate_decision.json").exists()


def test_main_prints_the_decision(tmp_path, tars, sm, monkeypatch, capsys):
    """The processing job's own command: python3 m13_03_evaluate_vs_champion.py, exit 0 and the decision line."""
    s3 = champion_in_registry(sm, tars["p3_01_baseline"])
    monkeypatch.setattr(m13, "sagemaker_client", lambda sm_=None: sm)
    monkeypatch.setattr(m13, "s3_client", lambda s3_=None: s3)
    layout(tmp_path, tars["p5_03_production"])
    assert m13.main(["--processing-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "PROMOTE (promote 1): candidate 36,975 > champion 19,225 (+17,750)" in out
    assert "champion: p3_01_baseline from package ccfd-fraud-detector/3 (Approved)" in out
    assert decision_file(tmp_path)["promote"] == 1


def test_cli_bad_input_exits_1(tmp_path):
    """No candidate in input/model: exit 1 with one FAILED line, before any AWS call."""
    need(GATE, "the gate holdout data/gate_holdout.csv")
    (tmp_path / "input" / "gate").mkdir(parents=True)
    shutil.copy(GATE, tmp_path / "input" / "gate" / "gate_holdout.csv")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    result = subprocess.run([sys.executable, "m13_03_evaluate_vs_champion.py", "--processing-dir", str(tmp_path)],
                            cwd=PROJECT, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "FAILED: candidate: no folder" in result.stdout


def test_newest_approved_follows_next_token_past_an_empty_page():
    """An empty first page with a NextToken is not 'no champion yet'."""
    import datetime

    import boto3
    from botocore.stub import Stubber

    import m13_03_evaluate_vs_champion as ev

    client = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response("describe_model_package_group", {
        "ModelPackageGroupName": ev.GROUP, "ModelPackageGroupArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:"
        "model-package-group/ccfd-fraud-detector", "CreationTime": datetime.datetime(2026, 10, 21), "CreatedBy": {}, "ModelPackageGroupStatus": "Completed"},
        {"ModelPackageGroupName": ev.GROUP})
    base = {"ModelPackageGroupName": ev.GROUP, "ModelApprovalStatus": "Approved", "SortBy": "CreationTime",
            "SortOrder": "Descending", "MaxResults": 10}
    stub.add_response("list_model_packages", {"ModelPackageSummaryList": [], "NextToken": "t1"}, base)
    arn = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/ccfd-fraud-detector/2"
    stub.add_response("list_model_packages", {"ModelPackageSummaryList": [{
        "ModelPackageArn": arn, "ModelPackageStatus": "Completed", "CreationTime": datetime.datetime(2026, 10, 21),
        "ModelApprovalStatus": "Approved"}]}, {**base, "NextToken": "t1"})
    with stub:
        assert ev.newest_approved(sm=client) == arn
