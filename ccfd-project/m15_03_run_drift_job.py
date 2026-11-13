"""
MLOps 15: run m15_03_drift_job.py as a processing job (ml.t3.medium, the training image) and wait.

    python m15_03_run_drift_job.py [--with-labels] [--publish-metrics]

Inputs: the reference (s3://<jobs>/reference/), the capture (s3://<jobs>/data-capture/ccfd-prod/) and, with
--with-labels, the late labels (s3://<jobs>/labels/). Output: s3://<jobs>/drift-reports/<job name>/.
"""
import argparse
import time

from mlops_utils import config, jobs

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--with-labels", action="store_true")
parser.add_argument("--publish-metrics", action="store_true")
args = parser.parse_args()
settings = config.load()
bucket = settings.bucket("jobs")
name = f"ccfd-drift-{time.strftime('%Y%m%d-%H%M%S')}"
inputs = [jobs.processing_input("reference", f"s3://{bucket}/reference/", "/opt/ml/processing/input/reference"),
          jobs.processing_input("capture", f"s3://{bucket}/data-capture/ccfd-prod/", "/opt/ml/processing/input/capture")]
if args.with_labels:
    inputs.append(jobs.processing_input("labels", f"s3://{bucket}/labels/", "/opt/ml/processing/input/labels"))
arguments = [flag for flag, on in (("--with-labels", args.with_labels), ("--publish-metrics", args.publish_metrics)) if on]
jobs.start_processing_job(name, settings.get("MLOPS_TRAINING_IMAGE_URI"), settings.get("MLOPS_EXECUTION_ROLE_ARN"),
                          ["python3", "/opt/program/m15_03_drift_job.py"], inputs,
                          [jobs.processing_output("report", f"s3://{bucket}/drift-reports/{name}/",
                                                  "/opt/ml/processing/output/report")],
                          arguments=arguments or None, tags=settings.aws_tags(), log=print)
jobs.wait_for_processing_job(name, log=print)
print(f"report: s3://{bucket}/drift-reports/{name}/drift_report.html")
