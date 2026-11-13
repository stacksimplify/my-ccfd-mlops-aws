"""
MLOps 05 Step-04: roll back (or forward) by moving an alias.

    python m05_04_move_alias.py --alias champion --to-version 1 --reason "rollback demo"

The alias moves; no version is deleted or changed. The version it moves to gets the tags rolled_back_at and
rollback_reason, so the registry shows who is serving and why.
"""

import argparse
from datetime import UTC, datetime

import m05_common
from mlops_utils import mlflow_layer

parser = argparse.ArgumentParser(description="Move an alias of ccfd-fraud-model to another version.")
parser.add_argument("--alias", default=mlflow_layer.CHAMPION, choices=[mlflow_layer.CHAMPION, mlflow_layer.CANDIDATE])
parser.add_argument("--to-version", required=True)
parser.add_argument("--reason", required=True)
args = parser.parse_args()

m05_common.connect()
before = mlflow_layer.set_alias(args.alias, args.to_version)
mlflow_layer.tag_version(args.to_version, {"rolled_back_at": datetime.now(UTC).isoformat(timespec="seconds"),
                                           "rollback_reason": args.reason})
print(f"@{args.alias}: version {before} -> version {args.to_version} (reason: {args.reason})")
