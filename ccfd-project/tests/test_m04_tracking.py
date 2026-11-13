"""MLOps 04: the m04 tracking helper (no server needed: a SQLite tracking store in a temp folder)."""
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest

import m04_tracking as track

PROJECT = Path(__file__).resolve().parents[1]


def test_connect_stops_when_the_tracking_uri_is_not_set(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(SystemExit, match="MLFLOW_TRACKING_URI is not set"):
        track.connect()


def test_arguments_has_the_lab_repo_and_the_data_tag_only(monkeypatch):
    monkeypatch.setattr("sys.argv", ["m04_03_track_gridsearch.py"])
    args = track.arguments("test")
    assert args.data_tag == "data-v1" and args.dvc_repo.name == "ccfd-project"
    assert set(vars(args)) == {"dvc_repo", "data_tag"}
    # The 300-fit grid search is never re-run: there is no --full mode
    monkeypatch.setattr("sys.argv", ["m04_03_track_gridsearch.py", "--full"])
    with pytest.raises(SystemExit):
        track.arguments("test")


def test_gridsearch_logs_the_frozen_settings_only():
    code = (PROJECT / "m04_03_track_gridsearch.py").read_text(encoding="utf-8")
    assert 'FROZEN_CONFIG = "config/p5_02_best_config.yaml"' in code
    assert "p5_02_class_weight_gridsearch.py\", \"gridsearch\"" in code   # the lineage tag names the base search
    assert "run_base(\"p5_02_class_weight_gridsearch.py\")" not in code     # the search itself never runs
    assert "--full" not in code and "full_flag" not in code


def test_no_m04_script_writes_a_tracking_uri():
    for path in PROJECT.glob("m04_*.py"):
        code = path.read_text(encoding="utf-8")
        assert not re.search(r"set_tracking_uri\(\s*['\"]", code), path.name


def test_pointer_md5_at_tag_reads_the_dvc_file_of_that_tag(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    (tmp_path / "data").mkdir()
    pointer = tmp_path / f"{track.DATA_FILE}.dvc"
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    pointer.write_text("outs:\n- md5: aaaa\n  size: 1\n  path: x.csv\n")
    git("add", "-A")
    git("commit", "-q", "-m", "v1")
    git("tag", "data-v1")
    pointer.write_text("outs:\n- md5: bbbb\n  size: 2\n  path: x.csv\n")
    git("commit", "-q", "-am", "v2")
    assert track.pointer_md5_at_tag(tmp_path, "data-v1") == "aaaa"
    with pytest.raises(SystemExit, match="no data/"):
        track.pointer_md5_at_tag(tmp_path, "data-v9")


def test_log_children_makes_one_nested_run_per_candidate(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    search = SimpleNamespace(cv_results_={
        "params": [{"model__C": 0.1}, {"model__C": 1.0}, {"model__C": 10.0}],
        "mean_test_score": [34070.0, 20900.0, -100.0], "rank_test_score": [1, 2, 3]})
    with mlflow.start_run() as parent:
        pass
    assert track.log_children(parent.info.run_id, search) == 3
    children = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 3
    assert sorted(children["metrics.cv_net_benefit"]) == [-100.0, 20900.0, 34070.0]
    assert set(children["params.C"]) == {"0.1", "1.0", "10.0"}
