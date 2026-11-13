"""
MLOps 09 Step-03: two registries, one story. Follow an Approved package back to its data and code, then point
MLflow @champion at the same model.

    python m09_03_lineage_report.py --version <n>

The chain comes from describe calls and the package metadata: package -> model.tar.gz -> training job -> its fe
input -> processing job -> its raw input (the DVC object, named by the data md5) -> git commit. Then @champion in
the MLflow App moves to the version registered from the training job's run (mlops_utils/release.py).
"""

import argparse

import m09_common as m09
from mlops_utils import endpoints, mlflow_layer, release

parser = argparse.ArgumentParser(description="Print the lineage of a package and sync MLflow @champion.")
parser.add_argument("--version", required=True, type=int)
parser.add_argument("--no-sync", action="store_true", help="print the lineage only")
args = parser.parse_args()

sm = m09.sm()
package = m09.package(args.version)
meta = package.get("CustomerMetadataProperties", {})
tar = package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
training = sm.describe_training_job(TrainingJobName=meta["training_job"])
fe = next(c for c in training["InputDataConfig"] if c["ChannelName"] == "fe")["DataSource"]["S3DataSource"]["S3Uri"]
processing = sm.describe_processing_job(ProcessingJobName=meta["processing_job"])
raw = next(i for i in processing["ProcessingInputs"] if i["InputName"] == "raw")["S3Input"]["S3Uri"]
md5 = raw.rsplit("/", 2)[-2] + raw.rsplit("/", 1)[-1]
if md5 != meta.get("dvc_md5"):
    raise SystemExit(f"STOP: the processing input md5 {md5} is not the package's dvc_md5 {meta.get('dvc_md5')}")

m09.say(f"{m09.GROUP}/{args.version} ({package['ModelApprovalStatus']}), net_benefit {meta.get('net_benefit')}")
m09.say(f"  -> model.tar.gz      {tar}")
m09.say(f"  -> training job      {training['TrainingJobName']} (spot {training.get('EnableManagedSpotTraining')}, "
        f"billable {training.get('BillableTimeInSeconds')} s)")
m09.say(f"  -> MLflow run        {meta.get('mlflow_run_id', 'none')} (ccfd-fraud-model version {meta.get('mlflow_version', 'none')})")
m09.say(f"  -> fe input          {fe}")
m09.say(f"  -> processing job    {processing['ProcessingJobName']}")
m09.say(f"  -> DVC data          md5 {md5} ({raw})")
m09.say(f"  -> code              git {meta.get('git_sha')}, image {meta.get('image_digest', '')[:19]}...")

if not args.no_sync:
    newest = endpoints.newest_approved(m09.GROUP)
    if newest is None or int(newest["ModelPackageVersion"]) != args.version:
        raise SystemExit("STOP: sync only the newest Approved package (it is the champion).")
    if meta.get("mlflow_version"):   # m09_01 registered the run already: point @champion at that version
        mlflow_layer.connect()
        mlflow_layer.tag_version(meta["mlflow_version"], {"package_version": args.version})
        before = mlflow_layer.set_alias(mlflow_layer.CHAMPION, meta["mlflow_version"])
        m09.say(f"@champion: version {before} -> {meta['mlflow_version']} (package {args.version})")
    else:
        release.sync_champion(newest["ModelPackageArn"], log=m09.say)
    synced = release.champion_package_version() == args.version
    m09.say(f"Registries in sync: {'YES' if synced else 'NO'}")
