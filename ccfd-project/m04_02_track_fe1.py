"""
MLOps 04 Step-02: the base project's FE1 model (p4_03, with amount_ratio) as the MLflow run 'p4_03-fe1'.

    python m04_02_track_fe1.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

Run the base project's p4_01_feature_engineering_fe1.py first (it writes the FE1 CSV). The base project's p4_03 runs unchanged.
"""

import m04_tracking as track

FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"

args = track.arguments("Track the base project's p4_03 FE1 model in MLflow.")
track.connect()
tags = track.data_tags("p4_03_train_model_with_fe1.py", "fe1", args)
base = track.run_base("p4_03_train_model_with_fe1.py")
track.track("p4_03-fe1", base["pipeline"], base["metrics"]["business_metrics"]["net_benefit"], tags, train_csv=FE_FILE)
