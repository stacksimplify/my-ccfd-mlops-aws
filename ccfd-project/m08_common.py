"""
MLOps 08: the shared steps of the m08 job scripts.

  settings()         the MLOPS_* values of .env.mlops (stop with the fix when one is missing)
  dvc_object(...)    the S3 URI of a data version's DVC object (md5 from the lab repo's pointer at a git tag)
  stamp()            a UTC time stamp for job names
  say(text)          print through the account ID mask
"""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mlops_utils import config

DATA_FILE = "data/credit_card_transactions_latest.csv"


def say(text: str) -> None:
    print(config.mask(str(text)), flush=True)


def settings(*keys: str) -> config.Settings:
    loaded = config.load()
    missing = [key for key in keys if not loaded.get(key)]
    if missing:
        sys.exit(f"STOP: .env.mlops has no {', '.join(missing)}. Run: bash ../scripts/create-env.sh --write-env-only")
    return loaded


def dvc_object(lab: Path, tag: str, suffix: str) -> tuple[str, str]:
    shown = subprocess.run(["git", "-C", str(lab), "show", f"{tag}:{DATA_FILE}.dvc"],
                           check=False, capture_output=True, text=True)
    if shown.returncode != 0:
        sys.exit(f"STOP: no {DATA_FILE}.dvc at tag {tag} in {lab} (MLOps 03).")
    md5 = yaml.safe_load(shown.stdout)["outs"][0]["md5"]
    return md5, f"s3://ccfd-mlops-dvc-{suffix}/dvc/files/md5/{md5[:2]}/{md5[2:]}"


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
