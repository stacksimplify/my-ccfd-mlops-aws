"""
MLOps 13: score the candidate and the champion on the fixed gate holdout, then decide.

The pipeline's EvaluateVsChampion step runs this file as a processing job in the training image (one jobs image):
    ContainerEntrypoint ["python3", "/opt/program/m13_03_evaluate_vs_champion.py"]   (no arguments)
What the job gives the container (the job request sets these folders):
  /opt/ml/processing/input/model/model.tar.gz   the candidate: the training step's model (model.tar.gz layout)
  /opt/ml/processing/input/gate/<one file>      the fixed gate holdout (GateHoldoutS3Uri: gate/gate_holdout.csv;
                                                MLOps 16 passes month_4 the same way)
What this file writes:
  /opt/ml/processing/output/gate/gate_decision.json
        both Net Benefits, the gain, the decision PROMOTE or REJECT and "promote": 1 or 0, the field the pipeline's
        condition step reads. Never evaluation.json: that is the model's own file.

The champion: the newest Approved package in ccfd-fraud-detector (list_model_packages); its ModelDataUrl (a
model.tar.gz in S3) is downloaded. No Approved package yet means there is no champion. The group itself must exist
(describe_model_package_group first): a missing group fails the job, so it never looks like "no champion yet".

The rule (mlops_utils/gate.py): both models are scored on the same gate data with the base project's Net Benefit, threshold
0.5 for both. PROMOTE (promote 1) only if the candidate is strictly better. Equal or lower is REJECT (promote 0), so a
tie keeps the champion. With no champion yet, the candidate is promoted (promote 1).
"""

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import traceback
import zlib
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config, gate

# The base project's code, unchanged
from utils.model_utils import load_pipeline

PROCESSING_DIR = Path("/opt/ml/processing")
DECISION_NAME = "gate_decision.json"
MODEL_FILES = ("model_latest.pkl", "metadata_latest.json", "evaluation.json")   # the tar root
GROUP = config.MODEL_PACKAGE_GROUP                                         # ccfd-fraud-detector


class EvaluateError(Exception):
    """A problem with the job's inputs: the message is the job's failure reason."""


def say(message: str) -> None:
    print(f"[m13_03] {config.mask(message)}", flush=True)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Files: the gate data and the 2 models
# ---------------------------------------------------------------------------
def single_file(source: Path, what: str) -> Path:
    """The one file in a processing input folder, whatever its name. Hidden files are skipped."""
    if not source.is_dir():
        raise EvaluateError(f"{what}: no folder {source}.")
    files = sorted(path for path in source.rglob("*") if path.is_file() and not path.name.startswith("."))
    if len(files) != 1:
        names = ", ".join(str(path.relative_to(source)) for path in files[:5]) or "none"
        raise EvaluateError(f"{what}: {source.name} must hold exactly 1 file, found {len(files)} ({names}).")
    return files[0]


def split_s3_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key or key.endswith("/"):
        raise EvaluateError(f"{uri} is not an s3://bucket/key URI of a model.tar.gz.")
    return bucket, key


def s3_client(s3=None):
    return s3 if s3 is not None else boto3.client("s3", region_name=config.REGION)


def sagemaker_client(sm=None):
    return sm if sm is not None else boto3.client("sagemaker", region_name=config.REGION)


def unpack(tar_path: Path, dest: Path) -> Path:
    """Unpack a model.tar.gz into dest. The 3 model files must sit at the tar root."""
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            names = {member.name.removeprefix("./") for member in tar.getmembers() if member.isfile()}
            missing = [name for name in MODEL_FILES if name not in names]
            if missing:
                raise EvaluateError(f"{tar_path.name} is not in the model.tar.gz layout: {', '.join(missing)} missing at "
                                    "the tar root.")
            dest.mkdir(parents=True, exist_ok=True)
            tar.extractall(dest, filter="data")
    except tarfile.TarError as error:
        raise EvaluateError(f"{tar_path.name} is not a model.tar.gz ({error}).") from error
    except (EOFError, OSError, zlib.error) as error:
        # For example a model.tar.gz cut short by a partial download: the gzip stream ends too early
        raise EvaluateError(f"{tar_path.name} is not a complete model.tar.gz ({type(error).__name__}: {error}).") \
            from error
    return dest


def download(uri: str, tar_path: Path, s3=None) -> Path:
    """The champion's model.tar.gz from S3 (its package's ModelDataUrl) into tar_path."""
    bucket, key = split_s3_uri(uri)
    s3_client(s3).download_file(bucket, key, str(tar_path))
    return tar_path


def load_model(folder: Path):
    """model_latest.pkl and metadata_latest.json with the base project's load_pipeline (as the serving API loads a model)."""
    with contextlib.redirect_stdout(io.StringIO()):     # the base project's loader prints emoji lines; keep our log ASCII
        pipeline, metadata = load_pipeline(folder.name, save_dir=str(folder.parent))
    return pipeline, metadata


def version_of(metadata: dict) -> str:
    """The base project's own version stamp of a model (metadata timestamp, as the original monitoring study used it)."""
    return str(metadata.get("timestamp") or metadata.get("created_at") or "unknown")


# ---------------------------------------------------------------------------
# 2. The champion from the SageMaker model registry
# ---------------------------------------------------------------------------
def newest_approved(sm=None) -> str | None:
    """
    The ARN of the newest Approved package in ccfd-fraud-detector, or None when the group has none yet.
    The group must exist: an empty list for a missing group would look like "no champion yet" and PROMOTE.
    """
    client = sagemaker_client(sm)
    try:
        client.describe_model_package_group(ModelPackageGroupName=GROUP)
    except ClientError as error:
        details = error.response.get("Error", {})
        raise EvaluateError(config.mask(
            f"The model package group {GROUP} cannot be read ({details.get('Code')}: {details.get('Message')}). "
            "A missing group is an error, never 'no champion yet': MLOps 09 creates it (after a full exit, see "
            "'Coming Back After a Full Exit' in MLOps 16)."
        )) from None
    # Follow NextToken: with a status filter a page can come back empty while more pages exist, and an empty first
    # page must never read as "no champion yet" (that would PROMOTE). Decision 10-x item 1, 2026-10-21.
    request = {"ModelPackageGroupName": GROUP, "ModelApprovalStatus": "Approved",
               "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 10}
    while True:
        answer = client.list_model_packages(**request)
        packages = answer.get("ModelPackageSummaryList", [])
        if packages:
            return packages[0]["ModelPackageArn"]
        if not answer.get("NextToken"):
            return None
        request["NextToken"] = answer["NextToken"]


def package_model(arn: str, sm=None) -> tuple[str, str]:
    """
    A package's model.tar.gz URI (InferenceSpecification.Containers[0].ModelDataUrl) and a short label such as
    'package ccfd-fraud-detector/3 (Approved)'. The label never holds the account ID, so the decision file can be
    shown on camera.
    """
    package = sagemaker_client(sm).describe_model_package(ModelPackageName=arn)
    containers = package.get("InferenceSpecification", {}).get("Containers", [])
    url = containers[0].get("ModelDataUrl") if containers else None
    group = package.get("ModelPackageGroupName", GROUP)
    label = f"package {group}/{package.get('ModelPackageVersion', '?')} ({package.get('ModelApprovalStatus')})"
    if not url:
        raise EvaluateError(f"{label} has no InferenceSpecification.Containers[0].ModelDataUrl, so its model cannot "
                            "be read.")
    return url, label


# ---------------------------------------------------------------------------
# 3. The job
# ---------------------------------------------------------------------------
def evaluate(processing_dir: Path = PROCESSING_DIR, *, sm=None, s3=None) -> dict:
    """
    Score the candidate and the champion on the gate data, decide with mlops_utils/gate.py and write
    <processing_dir>/output/gate/gate_decision.json. Returns the decision record.
    """
    gate_file = single_file(processing_dir / "input" / "gate", "gate data")
    try:
        data = gate.load_gate_data(gate_file)
    except (KeyError, ValueError, UnicodeDecodeError) as error:     # pandas' parser errors are ValueErrors
        raise EvaluateError(f"gate data: {gate_file.name} cannot be read as the base project's data "
                            f"({type(error).__name__}: {error}).") from error
    if gate.TARGET not in data.columns:
        raise EvaluateError(f"gate data: {gate_file.name} has no {gate.TARGET} column, so no Net Benefit can be "
                            "computed.")
    if data.empty:
        raise EvaluateError(f"gate data: {gate_file.name} has a header but no rows.")
    say(f"gate data: {gate_file.name} ({len(data):,} rows, {int(data[gate.TARGET].sum())} frauds, "
        f"md5 {md5(gate_file)})")

    with tempfile.TemporaryDirectory(prefix="mlops-gate-") as tmp:
        work = Path(tmp)
        candidate_tar = single_file(processing_dir / "input" / "model", "candidate")
        candidate_model, candidate_meta = load_model(unpack(candidate_tar, work / "candidate"))
        candidate_source = f"file {candidate_tar.name}"
        say(f"candidate: {candidate_meta.get('model_name')} from {candidate_source}")

        champion_arn = newest_approved(sm)
        champion_model = champion_meta = champion_source = None
        if champion_arn:
            choice = f"the newest Approved package in {GROUP}"
            url, champion_source = package_model(champion_arn, sm)
            champion_tar = download(url, work / "champion.tar.gz", s3)
            champion_model, champion_meta = load_model(unpack(champion_tar, work / "champion"))
            say(f"champion: {champion_meta.get('model_name')} from {champion_source} (chosen by {choice})")
        else:
            choice = f"none: no Approved package in {GROUP} yet, so the candidate is the first champion"
            say(f"champion: {choice}")

        record = gate.run_gate(
            candidate_model, champion_model, data,
            candidate_name=candidate_meta.get("model_name"), candidate_version=version_of(candidate_meta),
            champion_name=champion_meta.get("model_name") if champion_meta else None,
            champion_version=version_of(champion_meta) if champion_meta else None,
            data_name=gate_file.name)

    record["candidate"]["source"] = candidate_source
    if record["champion"] is not None:
        record["champion"]["source"] = champion_source
    record["champion_choice"] = choice
    record["gate_data_md5"] = md5(gate_file)
    path = gate.write_decision(record, processing_dir / "output" / "gate" / DECISION_NAME)
    say(f"{record['decision']} (promote {record['promote']}): {record['reason']}")
    say(f"decision: {path.name} -> {path.parent}")
    summary = {"decision": record["decision"], "promote": record["promote"],
               "candidate_net_benefit": record["candidate"]["net_benefit"],
               "champion_net_benefit": record["champion"]["net_benefit"] if record["champion"] else None,
               "net_benefit_gain": record["net_benefit_gain"], "gate_data": record["gate_data"]}
    say("GATE SUMMARY " + json.dumps(summary))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate: candidate vs champion on the fixed gate holdout.")
    parser.add_argument("--processing-dir", type=Path, default=PROCESSING_DIR,
                        help="the processing folder (default /opt/ml/processing); the decision goes to <it>/output/gate/")
    args = parser.parse_args(argv)
    try:
        evaluate(args.processing_dir)
    except EvaluateError as error:
        say(f"FAILED: {error}")
        return 1
    except Exception as error:
        # For example an AWS error: its text can hold an ARN with the account ID, so the traceback and the
        # message are masked; SystemExit(1) then ends the job without printing the unmasked traceback
        print(config.mask(traceback.format_exc()), file=sys.stderr, flush=True)
        say(f"FAILED: {type(error).__name__}: {error}")
        raise SystemExit(1) from error
    return 0


if __name__ == "__main__":
    sys.exit(main())
