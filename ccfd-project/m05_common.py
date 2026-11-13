"""
MLOps 05: the shared steps of the m05 scripts.

  connect()          MLFLOW_TRACKING_URI only (exit 1 when it is not set)
  stage_run(stage)   the newest MLOps 04 parent run with that journey_stage tag; when it is missing, say which
                     MLOps 04 script makes it and exit 1
"""

import os
import sys

from mlops_utils import mlflow_layer

MAKES = {"baseline": "m04_02_track_baseline.py", "fe1": "m04_02_track_fe1.py",
         "randomsearch": "m04_03_track_randomsearch.py", "gridsearch": "m04_03_track_gridsearch.py",
         "production": "m04_04_track_production.py"}


def connect() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        sys.exit("STOP: MLFLOW_TRACKING_URI is not set. Run: export MLFLOW_TRACKING_URI=http://127.0.0.1:7080")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")   # no download bars in the output
    # A server that does not answer stops with one STOP line, never a traceback
    try:
        return mlflow_layer.connect(uri)
    except RuntimeError as error:
        sys.exit(f"STOP: {error}")


def stage_run(stage: str):
    run = mlflow_layer.latest_run("journey_stage", stage)
    if run is None:
        print(f"STOP: no MLOps 04 run with journey_stage={stage}. Run MLOps 04 first ({MAKES[stage]}).")
        sys.exit(1)
    return run
