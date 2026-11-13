"""
MLOps 05 Step-02: the Net Benefit gate, replayed in the order MLOps 04 made the models (champion: FE1, version 1).

    python m05_02_net_benefit_gate.py --candidate-stage randomsearch    (run 1: version 2, REJECT -2,675)
    python m05_02_net_benefit_gate.py --candidate-stage production      (run 2: version 3, PROMOTE +10,850)

The newest MLOps 04 run of that stage is registered as a new version and gets the alias @candidate. Both
@champion and @candidate are scored on one fixed test set: the 2,000 held-out test rows of the base project's FE1 data
(split seed 42, 60 frauds, threshold 0.5), rows neither model trained on. PROMOTE only if the candidate's Net Benefit is
strictly higher; then @champion moves to it.
The decision is written as tags on the candidate version. No version is ever deleted. No F1 rule: Net Benefit prices a
missed fraud and a false alarm the way the business does.
"""

import argparse
from pathlib import Path

import m05_common
from mlops_utils import gate, mlflow_layer

parser = argparse.ArgumentParser(description="Gate one MLOps 04 run against @champion on Net Benefit.")
parser.add_argument("--candidate-stage", required=True,
                    choices=["baseline", "fe1", "randomsearch", "gridsearch", "production"])
stage = parser.parse_args().candidate_stage

if not Path(gate.BASE_FE_FILE).is_file():
    raise SystemExit(f"STOP: {gate.BASE_FE_FILE} not found. Run: python p1_01_generate_initial_dataset.py && "
                     "python p4_01_feature_engineering_fe1.py")
data = gate.base_test_set()   # the fixed gate data, read before anything is registered
m05_common.connect()
champion = mlflow_layer.get_alias(mlflow_layer.CHAMPION)
if champion is None:
    raise SystemExit("STOP: no @champion yet. Run m05_01_register_first_champion.py first.")
run = m05_common.stage_run(stage)
candidate = mlflow_layer.register(run.info.run_id, {"journey_stage": stage, "data_md5": run.data.tags.get("data_md5")})
mlflow_layer.set_alias(mlflow_layer.CANDIDATE, candidate.version)

record = gate.run_gate(mlflow_layer.load_version(candidate.version), mlflow_layer.load_version(champion.version),
                       data, candidate_name=run.info.run_name, candidate_version=candidate.version,
                       champion_name=mlflow_layer.MODEL_NAME, champion_version=champion.version,
                       data_name="The base project test split (FE1, seed 42)")
mlflow_layer.tag_version(candidate.version, gate.version_tags(record))

print(f"Gate data: {record['gate_rows']:,} rows, {record['gate_frauds']} frauds, threshold {gate.THRESHOLD}")
print(f"@champion  version {champion.version}: ${record['champion']['net_benefit']:,}")
print(f"@candidate version {candidate.version}: ${record['candidate']['net_benefit']:,} ({run.info.run_name})")
print(f"Gate: {record['decision']} ({record['net_benefit_gain']:+,})")
if record["promote"]:
    mlflow_layer.set_alias(mlflow_layer.CHAMPION, candidate.version)
    print(f"@champion -> version {candidate.version}")
else:
    print(f"@champion stays on version {champion.version}")
