"""
MLOps 04 Step-03: the base project's random search (p5_01) as the parent run 'p5_01-randomsearch' with one child run per
candidate (20).

    python m04_03_track_randomsearch.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

The base project's p5_01 runs unchanged (n_iter 20, cv 5, seed 42, the Net Benefit scorer).
"""

import m04_tracking as track

FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"

args = track.arguments("Track the base project's p5_01 random search in MLflow.")
track.connect()
tags = track.data_tags("p5_01_train_hyperparameter_tuning.py", "randomsearch", args)
base = track.run_base("p5_01_train_hyperparameter_tuning.py")
search = base["random_search"]
run_id = track.track("p5_01-randomsearch", base["best_pipeline"], base["metrics"]["business_metrics"]["net_benefit"],
                     {**tags, "best_cv_net_benefit": f"{search.best_score_:.0f}"}, train_csv=FE_FILE,
                     params=base["best_params_clean"])
print(f"Child runs: {track.log_children(run_id, search)} (best CV Net Benefit ${search.best_score_:,.0f})")
