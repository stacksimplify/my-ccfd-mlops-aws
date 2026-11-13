"""
MLOps 13: the SageMaker Pipeline ccfd-retraining-pipeline as a definition JSON.

Built from plain Python dicts (no SageMaker SDK), in the shape that ran on real AWS. The file
written here holds placeholders, never an account ID or a bucket suffix:
    <EXECUTION_ROLE_ARN>  <JOBS_BUCKET>  <MLFLOW_APP_ARN>  <TAG_OWNER>  <TAG_PURPOSE>  <TAG_CREATED>
m13_01_start_pipeline.py fills them from .env.mlops (in CI: from the repository secrets and variables) and creates or
updates the pipeline.

The 6 steps:
  DataChecksAndFE      Processing, ml.t3.medium: m08_02 (data checks + amount_ratio) on InputDataS3Uri    (cached)
  ProductionTrain      Training, managed spot ml.m5.large: the base project's p5_03 behind m08_03, channels fe + config,
                       Profiler off  (cached)
  EvaluateVsChampion   Processing: m13_03 scores the candidate and the champion (the newest Approved package) on
                       GateHoldoutS3Uri (never cached: the champion can change); writes gate_decision.json (property
                       file GateDecision)
  CheckBeatsChampion   Condition: GateDecision.promote == 1 (strictly better; a tie writes 0)
  RegisterCandidate    RegisterModel in ccfd-fraud-detector, status from ModelApprovalStatus (PendingManualApproval)
  CandidateNotBetter   Fail: 'REJECT: <the gate's reason>'

Run it from the ccfd-project folder:
    python m13_01_build_pipeline.py                     writes pipeline/ccfd-retraining-pipeline.json
    python m13_01_build_pipeline.py --print-definition  prints the JSON (with placeholders)
"""

import argparse
import json
from pathlib import Path

from mlops_utils import jobs

PIPELINE_NAME = "ccfd-retraining-pipeline"
GROUP = "ccfd-fraud-detector"
OUTPUT_FILE = Path("pipeline") / f"{PIPELINE_NAME}.json"
JOBS = "s3://<JOBS_BUCKET>"
TAGS = [{"Key": "course", "Value": "mlops-on-aws"}, {"Key": "owner", "Value": "<TAG_OWNER>"},
        {"Key": "purpose", "Value": "<TAG_PURPOSE>"}, {"Key": "created", "Value": "<TAG_CREATED>"}]
CACHE = {"Enabled": True, "ExpireAfter": "P30D"}

# The 8 parameters MLOps 13 teaches, in this order
PARAMETERS = [
    ("InputDataS3Uri", ""),                                       # the DVC object of a data version (--dvc-repo)
    ("DataVersion", "data-v1"),
    ("GitSha", "unknown"),
    ("ServingImageUri", ""),
    ("TrainingImageUri", ""),
    ("GateHoldoutS3Uri", f"{JOBS}/gate/gate_holdout.csv"),         # the fixed gate holdout (Step-00-07)
    ("TrainingConfigS3Uri", f"{JOBS}/config/p5_02_best_config.yaml"),  # the base project's fixed YAML
    ("ModelApprovalStatus", "PendingManualApproval"),
]


def param(name):
    return {"Get": f"Parameters.{name}"}


def execution_path(*parts):
    """s3://<JOBS_BUCKET>/pipeline/<execution id>/<parts>: each run keeps its own outputs."""
    return {"Std:Join": {"On": "/", "Values": [f"{JOBS}/pipeline", {"Get": "Execution.PipelineExecutionId"},
                                                *parts]}}


def s3_input(name, uri, local_path):
    return {"InputName": name, "S3Input": {"S3Uri": uri, "LocalPath": local_path, "S3DataType": "S3Prefix",
                                           "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated",
                                           "S3CompressionType": "None"}}


def s3_output(name, local_path):
    return {"OutputName": name, "S3Output": {"S3Uri": execution_path(name), "LocalPath": local_path,
                                             "S3UploadMode": "EndOfJob"}}


def processing(entrypoint, inputs, outputs):
    app = {"ImageUri": param("TrainingImageUri"), "ContainerEntrypoint": ["python3", f"/opt/program/{entrypoint}"]}
    return {
        "ProcessingResources": {"ClusterConfig": {"InstanceType": jobs.PROCESSING_INSTANCE, "InstanceCount": 1,
                                                  "VolumeSizeInGB": jobs.PROCESSING_VOLUME_GB}},
        "AppSpecification": app,
        "RoleArn": "<EXECUTION_ROLE_ARN>",
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {"Outputs": outputs},
        "StoppingCondition": {"MaxRuntimeInSeconds": jobs.PROCESSING_MAX_RUNTIME},
        "Tags": TAGS,
    }


def channel(name, uri):
    return {"ChannelName": name, "InputMode": "File", "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}


def gate(path):
    return {"Std:JsonGet": {"PropertyFile": {"Get": "Steps.EvaluateVsChampion.PropertyFiles.GateDecision"},
                            "Path": path}}


def as_text(value):
    """A pipeline value as a string (Std:Join of one item)."""
    return {"Std:Join": {"On": "", "Values": [value]}}


MODEL_ARTIFACT = {"Get": "Steps.ProductionTrain.ModelArtifacts.S3ModelArtifacts"}


def build_definition() -> dict:
    data_checks = {
        "Name": "DataChecksAndFE", "Type": "Processing", "CacheConfig": CACHE,
        "Arguments": processing("m08_02_data_checks.py",
                                [s3_input("raw", param("InputDataS3Uri"), "/opt/ml/processing/input/raw")],
                                [s3_output("fe", "/opt/ml/processing/output/fe"),
                                 s3_output("report", "/opt/ml/processing/output/report")]),
    }
    train = {
        "Name": "ProductionTrain", "Type": "Training", "CacheConfig": CACHE,
        "Arguments": {     # exactly as mlops_utils/jobs.py training_job_request()
            "AlgorithmSpecification": {"TrainingImage": param("TrainingImageUri"), "TrainingInputMode": "File"},
            "RoleArn": "<EXECUTION_ROLE_ARN>",
            "HyperParameters": {"stage": jobs.TRAINING_STAGE},
            "InputDataConfig": [
                channel("fe", {"Get": "Steps.DataChecksAndFE.ProcessingOutputConfig.Outputs['fe'].S3Output.S3Uri"}),
                channel("config", param("TrainingConfigS3Uri"))],
            "OutputDataConfig": {"S3OutputPath": f"{JOBS}/pipeline-models/"},
            "ResourceConfig": {"InstanceType": jobs.TRAINING_INSTANCE, "InstanceCount": 1,
                               "VolumeSizeInGB": jobs.TRAINING_VOLUME_GB},
            "EnableManagedSpotTraining": True,
            "StoppingCondition": {"MaxRuntimeInSeconds": jobs.TRAINING_MAX_RUNTIME,
                                  "MaxWaitTimeInSeconds": jobs.TRAINING_MAX_WAIT},
            "ProfilerConfig": dict(jobs.PROFILER_OFF),
            "Environment": {"MLFLOW_TRACKING_URI": "<MLFLOW_APP_ARN>"},   # removed when there is no App
            "Tags": TAGS,
        },
    }
    evaluate = {
        "Name": "EvaluateVsChampion", "Type": "Processing",
        "Arguments": processing(
            "m13_03_evaluate_vs_champion.py",
            [s3_input("model", MODEL_ARTIFACT, "/opt/ml/processing/input/model"),
             s3_input("gateholdout", param("GateHoldoutS3Uri"), "/opt/ml/processing/input/gate")],  # names unique per job
            [s3_output("gate", "/opt/ml/processing/output/gate")]),
        "PropertyFiles": [{"PropertyFileName": "GateDecision", "OutputName": "gate",
                           "FilePath": "gate_decision.json"}],
    }
    register = {
        "Name": "RegisterCandidate", "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": GROUP,
            "InferenceSpecification": {
                "Containers": [{"Image": param("ServingImageUri"), "ModelDataUrl": MODEL_ARTIFACT}],
                "SupportedContentTypes": ["application/json"], "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"]},
            "ModelApprovalStatus": param("ModelApprovalStatus"),
            "CustomerMetadataProperties": {   # lineage: the data, the code, the gate's numbers
                "data_version": param("DataVersion"), "input_data": param("InputDataS3Uri"),
                "git_sha": param("GitSha"), "made_by": PIPELINE_NAME,
                "execution": {"Get": "Execution.PipelineExecutionId"},
                # Values must be strings of letters, digits, spaces and _ . : / = + - @ only (AWS refuses an Integer and
                # the gate's reason text with "(" "," ">"); the reason stays in gate_decision.json
                "net_benefit": as_text(gate("candidate.net_benefit")), "gate_decision": gate("decision")},
        },
    }
    not_better = {"Name": "CandidateNotBetter", "Type": "Fail",
                  "Arguments": {"ErrorMessage": {"Std:Join": {"On": "", "Values": ["REJECT: ", gate("reason")]}}}}
    check = {
        "Name": "CheckBeatsChampion", "Type": "Condition",
        "Arguments": {"Conditions": [{"Type": "Equals", "LeftValue": gate("promote"), "RightValue": 1}],
                      "IfSteps": [register], "ElseSteps": [not_better]},
    }
    return {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": n, "Type": "String", "DefaultValue": d} for n, d in PARAMETERS],
        "Steps": [data_checks, train, evaluate, check],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the ccfd-retraining-pipeline definition (placeholders).")
    parser.add_argument("--print-definition", action="store_true", help="print the JSON instead of writing it")
    args = parser.parse_args(argv)
    text = json.dumps(build_definition(), indent=2) + "\n"
    if args.print_definition:
        print(text, end="")
        return 0
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(text)
    print(f"wrote {OUTPUT_FILE} ({len(build_definition()['Steps'])} top-level steps, {len(PARAMETERS)} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
