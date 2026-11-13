"""
MLOps 04 Step-02: the base project's baseline (p3_01) as the MLflow run 'p3_01-baseline'.

    python m04_02_track_baseline.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

The base project's p3_01_train_model_baseline.py runs unchanged (split seed 42, threshold 0.5); this script logs its model,
metrics, data and lineage to MLflow (MLFLOW_TRACKING_URI).
"""

import m04_tracking as track

args = track.arguments("Track the base project's p3_01 baseline in MLflow.")
track.connect()
tags = track.data_tags("p3_01_train_model_baseline.py", "baseline", args)
base = track.run_base("p3_01_train_model_baseline.py")
track.track("p3_01-baseline", base["pipeline"], base["metrics"]["business_metrics"]["net_benefit"], tags,
            train_csv=track.DATA_FILE)
