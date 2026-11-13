"""MLOps 13: the ccfd-retraining-pipeline definition (m13_01_build_pipeline.py) and its filling."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import m13_01_build_pipeline as builder
import m13_01_start_pipeline as starter

STEP_NAMES = ["DataChecksAndFE", "ProductionTrain", "EvaluateVsChampion", "CheckBeatsChampion",
              "RegisterCandidate", "CandidateNotBetter"]
PARAMETER_NAMES = ["InputDataS3Uri", "DataVersion", "GitSha", "ServingImageUri", "TrainingImageUri",
                   "GateHoldoutS3Uri", "TrainingConfigS3Uri", "ModelApprovalStatus"]


def all_steps(steps):
    for step in steps:
        yield step
        yield from all_steps(step.get("Arguments", {}).get("IfSteps", []))
        yield from all_steps(step.get("Arguments", {}).get("ElseSteps", []))


def by_name():
    return {s["Name"]: s for s in all_steps(builder.build_definition()["Steps"])}


def test_the_six_steps_and_eight_parameters():
    definition = builder.build_definition()
    assert [s["Name"] for s in all_steps(definition["Steps"])] == STEP_NAMES
    assert [p["Name"] for p in definition["Parameters"]] == PARAMETER_NAMES
    assert definition["Version"] == "2020-12-01"
    assert "PipelineExperimentConfig" not in definition


def test_evaluate_step_takes_no_arguments():
    """The gate's champion is always the newest Approved package: the job gets no ContainerArguments."""
    app = by_name()["EvaluateVsChampion"]["Arguments"]["AppSpecification"]
    assert app["ContainerEntrypoint"] == ["python3", "/opt/program/m13_03_evaluate_vs_champion.py"]
    assert "ContainerArguments" not in app


def test_condition_reads_promote_equals_one():
    condition = by_name()["CheckBeatsChampion"]["Arguments"]["Conditions"][0]
    assert condition["Type"] == "Equals" and condition["RightValue"] == 1
    assert condition["LeftValue"]["Std:JsonGet"]["Path"] == "promote"
    assert by_name()["CandidateNotBetter"]["Type"] == "Fail"


def test_cache_only_on_data_checks_and_training():
    steps = by_name()
    assert steps["DataChecksAndFE"]["CacheConfig"] == {"Enabled": True, "ExpireAfter": "P30D"}
    assert steps["ProductionTrain"]["CacheConfig"] == {"Enabled": True, "ExpireAfter": "P30D"}
    assert "CacheConfig" not in steps["EvaluateVsChampion"]


def test_training_step_is_block_b():
    train = by_name()["ProductionTrain"]["Arguments"]
    assert train["ProfilerConfig"] == {"DisableProfiler": True}
    assert train["EnableManagedSpotTraining"] is True
    assert train["HyperParameters"] == {"stage": "production"}
    assert train["ResourceConfig"]["InstanceType"] == "ml.m5.large"
    assert [c["ChannelName"] for c in train["InputDataConfig"]] == ["fe", "config"]
    for forbidden in ("DebugHookConfig", "DebugRuleConfigurations", "ProfilerRuleConfigurations", "TensorBoardOutputConfig"):
        assert forbidden not in train


def test_tags_key_value_and_no_account_or_suffix():
    text = json.dumps(builder.build_definition())
    assert '"key"' not in text and '"value"' not in text
    assert not re.search(r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])", text)
    assert "ccfd-mlops-jobs-" not in text and "<JOBS_BUCKET>" in text and "<EXECUTION_ROLE_ARN>" in text


def test_committed_json_matches_the_builder():
    committed = json.loads(Path(builder.OUTPUT_FILE).read_text())
    assert committed == builder.build_definition()


class FakeSettings:
    def __init__(self, app):
        self.values = {"MLOPS_EXECUTION_ROLE_ARN": "arn:aws:iam::ACCOUNT:role/ccfd-sagemaker-execution-role",
                       "MLOPS_MLFLOW_APP_ARN": app}
        self.tags = {"owner": "kalyan", "purpose": "test", "created": "2026-10-21"}

    def get(self, key):
        return self.values.get(key)

    def bucket(self, purpose):
        return f"ccfd-mlops-{purpose}-abc123"


def test_filled_definition_has_no_placeholder_and_drops_mlflow_without_app():
    text = starter.filled_definition(FakeSettings(app=None))
    assert "<" not in text.replace("<=", "") and "ccfd-mlops-jobs-abc123" in text
    train = next(s for s in json.loads(text)["Steps"] if s["Name"] == "ProductionTrain")
    assert "Environment" not in train["Arguments"]
    with_app = starter.filled_definition(FakeSettings(app="arn:aws:sagemaker:us-east-1:ACCOUNT:mlflow-app/app-X"))
    train = next(s for s in json.loads(with_app)["Steps"] if s["Name"] == "ProductionTrain")
    assert train["Arguments"]["Environment"]["MLFLOW_TRACKING_URI"].endswith("mlflow-app/app-X")


class FakeSageMaker:
    """Stands in for the boto3 client in wait(): each check returns the next state of the run.
    Like the API, the steps come newest first unless SortOrder="Ascending" is asked for."""

    def __init__(self, checks):
        self.checks = list(checks)            # [(run status, [(step name, step status), ...] oldest first), ...]
        self.current = None
        self.sort_orders = []

    def describe_pipeline_execution(self, PipelineExecutionArn):
        self.current = self.checks.pop(0)
        return {"PipelineExecutionStatus": self.current[0]}

    def list_pipeline_execution_steps(self, PipelineExecutionArn, SortOrder="Descending"):
        self.sort_orders.append(SortOrder)
        steps = [{"StepName": name, "StepStatus": status} for name, status in self.current[1]]
        return {"PipelineExecutionSteps": steps if SortOrder == "Ascending" else steps[::-1]}


def test_wait_prints_the_steps_oldest_first(monkeypatch, capsys):
    monkeypatch.setattr(starter.time, "sleep", lambda seconds: None)
    sm = FakeSageMaker([
        ("Executing", [("DataChecksAndFE", "Executing")]),
        ("Executing", [("DataChecksAndFE", "Succeeded"), ("ProductionTrain", "Executing")]),
        ("Executing", [("DataChecksAndFE", "Succeeded"), ("ProductionTrain", "Succeeded"),
                       ("EvaluateVsChampion", "Executing")]),
        ("Failed", [("DataChecksAndFE", "Succeeded"), ("ProductionTrain", "Succeeded"),
                    ("EvaluateVsChampion", "Succeeded"), ("CheckBeatsChampion", "Succeeded"),
                    ("CandidateNotBetter", "Failed")]),
    ])
    assert starter.wait(sm, "arn:aws:sagemaker:us-east-1:ACCOUNT:pipeline/p/execution/x") == "Failed"
    assert set(sm.sort_orders) == {"Ascending"}
    printed = [line.split(" ", 1)[1].strip() for line in capsys.readouterr().out.splitlines()]
    assert printed == ["DataChecksAndFE: Executing", "DataChecksAndFE: Succeeded", "ProductionTrain: Executing",
                       "ProductionTrain: Succeeded", "EvaluateVsChampion: Executing", "EvaluateVsChampion: Succeeded",
                       "CheckBeatsChampion: Succeeded", "CandidateNotBetter: Failed", "run ended: Failed"]


def test_start_passes_only_the_run_values(monkeypatch):
    """The other pipeline parameters keep their defaults (no --gate-holdout or --champion-package-arn flag)."""
    calls = []

    class FakeStart:
        def start_pipeline_execution(self, **kwargs):
            calls.append(kwargs)
            return {"PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:ACCOUNT:pipeline/p/execution/abc"}

    monkeypatch.setattr(starter, "data_uri", lambda *args: "s3://ccfd-mlops-dvc-abc123/dvc/files/md5/65/ce97")
    monkeypatch.setattr(starter, "git_sha", lambda: "1234567")
    settings = FakeSettings(app=None)
    settings.values.update({"MLOPS_SERVING_IMAGE_URI": "serving:sha-1", "MLOPS_TRAINING_IMAGE_URI": "training:sha-1"})
    args = SimpleNamespace(dvc_repo="lab", data_version="data-v1", pointer_file=None)
    starter.start(FakeStart(), settings, args)
    names = [p["Name"] for p in calls[0]["PipelineParameters"]]
    assert names == ["InputDataS3Uri", "DataVersion", "GitSha", "ServingImageUri", "TrainingImageUri"]
