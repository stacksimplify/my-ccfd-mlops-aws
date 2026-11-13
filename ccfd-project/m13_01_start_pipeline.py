"""
MLOps 13: create or update ccfd-retraining-pipeline, then start a run (boto3 only).

    python m13_01_start_pipeline.py --upsert                       create or update the pipeline from the definition
    python m13_01_start_pipeline.py --start --data-version data-v2 start a run on that data version of the lab repo
    python m13_01_start_pipeline.py --upsert --start --data-version data-v1 --wait

The definition comes from m13_01_build_pipeline.py; its placeholders are filled from .env.mlops (MLOPS_EXECUTION_ROLE_ARN,
MLOPS_JOBS_BUCKET or the suffix, MLOPS_MLFLOW_APP_ARN, the course tag values). In CI (ct.yml) the same values come from the
repository secrets and variables, and ct.yml only starts the pipeline: create and update run from here.
--data-version reads the .dvc pointer of that git tag in the lab repo (--dvc-repo, default ~/mlops-lab/ccfd-project) and
passes the DVC 3 object in the S3 remote as InputDataS3Uri.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3

import m13_01_build_pipeline as builder
from mlops_utils import config

POINTER = "data/credit_card_transactions_latest.csv.dvc"


def say(text):
    print(time.strftime("%H:%M:%S"), config.mask(str(text)), flush=True)


def filled_definition(settings) -> str:
    """The definition JSON with every placeholder replaced; stops when a needed value is missing."""
    values = {"<EXECUTION_ROLE_ARN>": settings.get("MLOPS_EXECUTION_ROLE_ARN"), "<JOBS_BUCKET>": settings.bucket("jobs"),
              "<TAG_OWNER>": settings.tags.get("owner"), "<TAG_PURPOSE>": settings.tags.get("purpose"),
              "<TAG_CREATED>": settings.tags.get("created")}
    missing = [k for k, v in values.items() if not v]
    if missing:
        sys.exit(f"STOP: no value for {', '.join(missing)}: run scripts/create-env.sh first (it writes .env.mlops)")
    definition = builder.build_definition()
    # The current images become the parameter defaults, so a start that passes no images (the 06-loop rule, ct.yml)
    # still runs the right ones (a real run showed it: the loop rule passes only the data, config and gate parameters)
    for parameter in definition["Parameters"]:
        image = {"ServingImageUri": "MLOPS_SERVING_IMAGE_URI", "TrainingImageUri": "MLOPS_TRAINING_IMAGE_URI"}.get(parameter["Name"])
        if image and settings.get(image):
            parameter["DefaultValue"] = settings.get(image)
    app_arn = settings.get("MLOPS_MLFLOW_APP_ARN")
    for step in definition["Steps"]:
        if step["Name"] == "ProductionTrain" and not app_arn:
            step["Arguments"].pop("Environment")          # no MLflow App: the job logs nothing (it says so)
    text = json.dumps(definition)
    for placeholder, value in {**values, "<MLFLOW_APP_ARN>": app_arn or ""}.items():
        text = text.replace(placeholder, value)
    assert "<" not in text.replace("<=", ""), "a placeholder was left in the definition"
    return text


def upsert(sm, settings) -> None:
    text, role = filled_definition(settings), settings.get("MLOPS_EXECUTION_ROLE_ARN")
    try:
        sm.describe_pipeline(PipelineName=builder.PIPELINE_NAME)
        sm.update_pipeline(PipelineName=builder.PIPELINE_NAME, PipelineDefinition=text, RoleArn=role)
        say(f"pipeline {builder.PIPELINE_NAME}: updated")
    except sm.exceptions.ResourceNotFound:
        sm.create_pipeline(PipelineName=builder.PIPELINE_NAME, PipelineDefinition=text, RoleArn=role,
                           Tags=settings.aws_tags())
        say(f"pipeline {builder.PIPELINE_NAME}: created")


def data_uri(dvc_repo: str, tag: str, settings, pointer_file: str | None = None) -> str:
    """The DVC 3 object of a data version: from a .dvc file (CI: the checked-out pointer) or a lab-repo git tag."""
    if pointer_file:
        pointer = Path(pointer_file).read_text()
    else:
        answer = subprocess.run(["git", "-C", dvc_repo, "show", f"{tag}:{POINTER}"], check=False,
                                capture_output=True, text=True)
        if answer.returncode != 0:
            # A missing lab repo or tag: one clear line instead of a traceback
            sys.exit(f"STOP: no {POINTER} at the tag {tag} in {dvc_repo}. The lab repo of MLOps 03 lives on your "
                     f"EC2 VM: run this there; its data tags: git -C {dvc_repo} tag")
        pointer = answer.stdout
    md5 = next(line.split(":", 1)[1].strip() for line in pointer.splitlines() if line.strip().startswith("- md5:"))
    return f"s3://{settings.bucket('dvc')}/dvc/files/md5/{md5[:2]}/{md5[2:]}"


def git_sha() -> str:
    answer = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return answer.stdout.strip() or "unknown"


def start(sm, settings, args) -> str:
    parameters = {"InputDataS3Uri": data_uri(args.dvc_repo, args.data_version, settings, args.pointer_file),
                  "DataVersion": args.data_version, "GitSha": git_sha(),
                  "ServingImageUri": settings.get("MLOPS_SERVING_IMAGE_URI"),
                  "TrainingImageUri": settings.get("MLOPS_TRAINING_IMAGE_URI")}
    answer = sm.start_pipeline_execution(
        PipelineName=builder.PIPELINE_NAME, PipelineExecutionDisplayName=f"{args.data_version}-{time.strftime('%H%M%S')}",
        PipelineParameters=[{"Name": k, "Value": v} for k, v in parameters.items()])
    say(f"started a run on {args.data_version}: {answer['PipelineExecutionArn'].rsplit('/', 1)[1]}")
    return answer["PipelineExecutionArn"]


def outputs(sm, arn, status) -> dict:
    """What cd.yml needs: promote true/false and the new package ARN (written to $GITHUB_OUTPUT in Actions)."""
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
    register = next((s for s in steps if s["StepName"] == "RegisterCandidate" and s["StepStatus"] == "Succeeded"), None)
    package = register["Metadata"]["RegisterModel"]["Arn"] if register else ""
    rejected = any(s["StepName"] == "CandidateNotBetter" for s in steps)
    result = {"status": status, "promote": "true" if package else "false",
              "package_version": package.rsplit("/", 1)[1] if package else "",
              "rejected": "true" if rejected else "false"}
    say(f"result: promote={result['promote']} rejected={result['rejected']} package version "
        f"{package.rsplit('/', 1)[1] if package else '-'}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write("".join(f"{k}={v}\n" for k, v in result.items()))
    return result


def wait(sm, arn) -> str:
    """Check the run every 30 seconds; print a step's line when its status changes, the oldest step first."""
    seen = {}
    while True:
        status = sm.describe_pipeline_execution(PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")
        for step in steps["PipelineExecutionSteps"]:
            cached = " (cache hit)" if step.get("CacheHitResult", {}).get("SourcePipelineExecutionArn") else ""
            line = f"{step['StepName']}: {step['StepStatus']}{cached} {step.get('FailureReason', '')}".strip()
            if seen.get(step["StepName"]) != line:
                say("  " + line)
                seen[step["StepName"]] = line
        if status in ("Succeeded", "Failed", "Stopped"):
            say(f"run ended: {status}")
            return status
        time.sleep(30)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create/update and start ccfd-retraining-pipeline.")
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--wait", action="store_true", help="follow the run until it ends")
    parser.add_argument("--data-version", default="data-v1")
    parser.add_argument("--dvc-repo", default=str(config.LAB_REPO))
    parser.add_argument("--pointer-file", default=None, help="a .dvc file to read the md5 from (CI)")
    args = parser.parse_args(argv)
    settings = config.load()
    sm = boto3.client("sagemaker", region_name=config.REGION)
    if args.upsert:
        upsert(sm, settings)
    if args.start:
        arn = start(sm, settings, args)
        if args.wait:
            result = outputs(sm, arn, wait(sm, arn))
            # PROMOTE (Succeeded) and REJECT (CandidateNotBetter) are both normal ends; anything else is an error
            return 0 if result["promote"] == "true" or result["rejected"] == "true" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
