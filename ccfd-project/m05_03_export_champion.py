"""
MLOps 05 Step-03: export @champion as the folder the serving image reads.

    python m05_03_export_champion.py [--dest ~/mlops-lab/models/champion]

The serving image has no MLflow inside, so the alias is resolved here, on your EC2 VM. The folder gets
model_latest.pkl, metadata_latest.json and evaluation.json; docker run mounts it at /opt/ml/model.
"""

import argparse
from pathlib import Path

import m05_common
from mlops_utils import mlflow_layer

parser = argparse.ArgumentParser(description="Export the @champion model as a MODEL_DIR folder.")
parser.add_argument("--dest", type=Path, default=Path.home() / "mlops-lab" / "models" / "champion")
dest = parser.parse_args().dest.expanduser()

m05_common.connect()
version = mlflow_layer.download_alias(mlflow_layer.CHAMPION, dest)
print(f"models:/{mlflow_layer.MODEL_NAME}@{mlflow_layer.CHAMPION} -> version {version.version}")
print(f"Wrote {dest}: {', '.join(sorted(p.name for p in dest.iterdir()))}")
