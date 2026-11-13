"""
MLOps 08 Step-03: training the tuned model on SageMaker, managed spot, logged to the MLflow App.

    python m08_03_run_training_job.py [--dvc-rev data-v1]

Channels: fe = the processing job's output (Step-02); config = the base project's fixed YAML in the jobs bucket
(config/p5_02_best_config.yaml, the same YAML as the laptop). The job runs the base project's p5_03 unchanged on one ml.m5.large
spot instance, writes the 3 model files into model.tar.gz and logs the run to the MLflow App (MLFLOW_TRACKING_URI from
.env.mlops). It waits and prints Training and Billable seconds and the spot saving.
"""

import argparse

import m08_common as m08
from mlops_utils import jobs

parser = argparse.ArgumentParser(description="Train the base project's tuned model as a SageMaker spot training job.")
parser.add_argument("--dvc-rev", default="data-v1")
args = parser.parse_args()

env = m08.settings("MLOPS_JOBS_BUCKET", "MLOPS_EXECUTION_ROLE_ARN", "MLOPS_TRAINING_IMAGE_URI", "MLOPS_MLFLOW_APP_ARN")
bucket, base = env.get("MLOPS_JOBS_BUCKET"), f"m08/{args.dvc_rev}"
name = f"ccfd-train-{args.dvc_rev.replace('data-', '')}-{m08.stamp()}"
jobs.start_training_job(name, env.get("MLOPS_TRAINING_IMAGE_URI"), env.get("MLOPS_EXECUTION_ROLE_ARN"),
                        fe_s3_uri=f"s3://{bucket}/{base}/fe/",
                        config_s3_uri=f"s3://{bucket}/config/p5_02_best_config.yaml",
                        output_s3_uri=f"s3://{bucket}/{base}/training/",
                        tags=env.aws_tags(), tracking_uri=env.get("MLOPS_MLFLOW_APP_ARN"), log=m08.say)
summary = jobs.wait_for_training_job(name, log=m08.say)
saving = 100 * (1 - summary["billable_seconds"] / max(summary["training_seconds"], 1))
m08.say(f"Managed spot saving: {saving:.0f}% (billable {summary['billable_seconds']} s of "
        f"{summary['training_seconds']} s)")
m08.say(f"Model: {summary['model_artifact']}")
