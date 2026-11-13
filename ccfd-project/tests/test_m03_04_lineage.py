"""MLOps 03 Step-02 and Step-04: the lab-repo maker runs no git or dvc, and the lineage record reads the .dvc md5."""
import json
import subprocess
from pathlib import Path

import m03_04_train_with_data_tag as m03_04

PROJECT = Path(__file__).resolve().parents[1]
MAKER = PROJECT.parent / "scripts" / "m03_02_make_lab_repo.sh"
CSV = "date,amount\n2026-01-01,10.0\n2026-01-02,20.0\n"


def make_lab(tmp_path: Path) -> Path:
    repo = tmp_path / "lab"
    (repo / "data").mkdir(parents=True)
    (repo / "artifacts").mkdir()
    (repo / m03_04.DATA_FILE).write_text(CSV)
    (repo / m03_04.CONFIG_FILE).write_text("C: 0.1\n")
    md5 = m03_04.md5_of(repo / m03_04.DATA_FILE)
    (repo / f"{m03_04.DATA_FILE}.dvc").write_text(
        f"outs:\n- md5: {md5}\n  size: {len(CSV)}\n  hash: md5\n  path: credit_card_transactions_latest.csv\n")
    return repo


def test_pointer_reads_md5_and_size(tmp_path):
    repo = make_lab(tmp_path)
    assert m03_04.pointer(repo) == {"md5": m03_04.md5_of(repo / m03_04.DATA_FILE), "size": len(CSV)}


def test_lineage_md5_equals_the_dvc_md5(tmp_path):
    repo = make_lab(tmp_path)
    record = m03_04.lineage(repo, 50550)
    assert record["data_md5"] == m03_04.pointer(repo)["md5"]
    assert record["data_rows"] == 2
    assert record["net_benefit"] == 50550
    assert record["git_tag"] == "(none)"   # not a git repo
    json.dumps(record)


def test_maker_copies_code_only_and_runs_no_git(tmp_path):
    target = tmp_path / "home" / "mlops-lab" / "ccfd-project"
    done = subprocess.run(["bash", str(MAKER), str(target)], check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (target / "p1_01_generate_initial_dataset.py").is_file()
    assert not (target / ".git").exists() and not (target / ".dvc").exists()
    assert not list(target.rglob("*.csv")) and not list(target.rglob("*.pkl"))


def test_maker_refuses_an_existing_target_and_a_target_inside_a_git_repo(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    assert subprocess.run(["bash", str(MAKER), str(existing)], check=False, capture_output=True).returncode == 1
    inside = PROJECT / "lab-should-not-be-made"
    done = subprocess.run(["bash", str(MAKER), str(inside)], check=False, capture_output=True, text=True)
    if subprocess.run(["git", "-C", str(PROJECT), "rev-parse"], check=False,
                      capture_output=True).returncode == 0:
        assert done.returncode == 1 and "inside the git repo" in done.stdout
        assert not inside.exists()
