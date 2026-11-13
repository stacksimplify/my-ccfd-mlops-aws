"""
MLOps 03 Step-04: train the tuned model on the data version DVC has checked out, and stamp that data
version on the model (its lineage tag).

    python m03_04_train_with_data_tag.py                         the lab repo ~/mlops-lab/ccfd-project
    python m03_04_train_with_data_tag.py --dvc-repo <folder>     another lab repo

It reads the md5 and the size of the data from DVC's pointer file data/credit_card_transactions_latest.csv.dvc,
checks that the CSV on disk is that version (same md5), copies the frozen settings of the base project's grid search
(config/p5_02_best_config.yaml -> artifacts/; the 300-fit grid search is not re-run: its winner can change with the
CPU, and it takes about 5 minutes on the course VM), and then runs the base project's own scripts, unchanged:

    p4_01_feature_engineering_fe1.py     the amount_ratio feature
    p5_03_train_production_model.py      the tuned model in models/p5_03_production/

The scripts' own output stays off the screen; when it holds a ConvergenceWarning (data v2: the training stops at its
iteration limit), it prints one line: WARNING: the training did not converge (max_iter reached).

Then it writes models/p5_03_production/lineage_latest.json: data_md5, data_rows, config_md5, git_commit, git_tag,
net_benefit and trained_at. The same data md5 later becomes the MLflow tag (MLOps 04), the S3 key (MLOps 06) and
the job input (MLOps 08 and 13).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

DATA_FILE = "data/credit_card_transactions_latest.csv"
CONFIG_FILE = "artifacts/p5_02_best_config.yaml"
MODEL_DIR = "models/p5_03_production"
FROZEN_CONFIG = "config/p5_02_best_config.yaml"
BASE_SCRIPTS = ["p4_01_feature_engineering_fe1.py", "p5_03_train_production_model.py"]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pointer(repo: Path) -> dict:
    """md5 and size from the .dvc file (a small YAML file: outs: - md5: ..., size: ..., path: ...)."""
    fields = {}
    for line in (repo / f"{DATA_FILE}.dvc").read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().lstrip("- ").partition(":")
        if key in ("md5", "size"):
            fields[key] = value.strip()
    return {"md5": fields["md5"], "size": int(fields["size"])}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def lineage(repo: Path, net_benefit: int) -> dict:
    data = repo / DATA_FILE
    with data.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1   # minus the header line
    return {
        "data_file": DATA_FILE,
        "data_md5": pointer(repo)["md5"],
        "data_rows": rows,
        "config_md5": md5_of(repo / CONFIG_FILE),
        "git_commit": git(repo, "rev-parse", "--short", "HEAD"),
        "git_tag": git(repo, "describe", "--tags", "--exact-match") or "(none)",
        "net_benefit": net_benefit,
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train on the checked-out DVC data version and record its lineage.")
    parser.add_argument("--dvc-repo", type=Path, default=Path.home() / "mlops-lab" / "ccfd-project")
    repo = parser.parse_args().dvc_repo.expanduser().resolve()

    if not (repo / f"{DATA_FILE}.dvc").is_file():
        sys.exit(f"STOP: {repo}/{DATA_FILE}.dvc not found. Run it in the lab repo (MLOps 03 Step-02).")
    want = pointer(repo)
    if not (repo / DATA_FILE).is_file() or md5_of(repo / DATA_FILE) != want["md5"]:
        sys.exit("STOP: the CSV on disk is not the version in the .dvc file. Run: dvc checkout")
    print(f"Data version: md5 {want['md5']} ({want['size']:,} bytes), tag {git(repo, 'describe', '--tags', '--always')}")

    (repo / CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
    (repo / CONFIG_FILE).write_bytes((repo / FROZEN_CONFIG).read_bytes())
    print(f"Settings: the frozen {FROZEN_CONFIG} (md5 {md5_of(repo / CONFIG_FILE)})")

    for script in BASE_SCRIPTS:
        print(f"Running the base project's {script} ...", flush=True)
        done = subprocess.run([sys.executable, script], cwd=repo, check=False, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-2000:], done.stderr[-2000:])
            sys.exit(f"STOP: {script} failed (its last lines are above).")
        # The base scripts' output stays off the screen, so a ConvergenceWarning would be hidden: say it in one line
        if "ConvergenceWarning" in done.stdout + done.stderr:
            print("WARNING: the training did not converge (max_iter reached)", flush=True)

    metadata = json.loads((repo / MODEL_DIR / "metadata_latest.json").read_text(encoding="utf-8"))
    record = lineage(repo, int(metadata["metrics"]["net_benefit"]))
    (repo / MODEL_DIR / "lineage_latest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MODEL_DIR}/lineage_latest.json:")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
