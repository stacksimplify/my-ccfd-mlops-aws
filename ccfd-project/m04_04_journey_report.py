"""
MLOps 04 Step-04: the base project's Improvement Journey, read back from MLflow (not from JSON files).

    python m04_04_journey_report.py

For each journey stage it takes the newest parent run with that journey_stage tag and prints its Net Benefit and
the change against the baseline.
"""

import sys

import m04_tracking as track
from mlops_utils import mlflow_layer

STAGES = [("baseline", "p3_01"), ("fe1", "p4_03"), ("randomsearch", "p5_01"), ("gridsearch", "p5_02"),
          ("production", "p5_03")]

track.connect()
rows = []
for stage, script in STAGES:
    run = mlflow_layer.latest_run("journey_stage", stage)
    if run is None:
        sys.exit(f"STOP: no run with journey_stage={stage}. Run the m04 script for {script} first.")
    rows.append((stage, script, int(run.data.metrics["net_benefit"]), run.info.run_id[:8]))

base = rows[0][2]
print("The base project's Improvement Journey, from MLflow")
print(f"{'stage':<14}{'script':<8}{'Net Benefit':>13}{'vs baseline':>14}  run")
for stage, script, value, run_id in rows:
    print(f"{stage:<14}{script:<8}{f'${value:,}':>13}{f'{value - base:+,}':>14}  {run_id}")
print(f"Total change: {rows[-1][2] - base:+,} USD")
