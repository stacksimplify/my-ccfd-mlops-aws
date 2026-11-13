"""
MLOps 03 and 11: the data generators make the base project's schema, and the same seed always makes the same file.

  m03_03_simulate_new_data.py   data versions and the gate holdout (MLOps 03, 09, 13)
  m11_02_generate_months.py     the 4 production months (MLOps 11, 15, 16)

Every run here writes into a temporary folder (pytest's tmp_path); the project's data/ folder is never written.
The commands, md5s and seeds of the data versions and of the gate holdout come from the course's data record,
manifest.yaml, next to ccfd-project/ or in it. Where there is none, the 3 tests that read it are skipped; with
MLOPS_REQUIRE_MODEL=1 (as in CI) they fail instead.

No 12-digit number is written in this file: the md5 of month_3 is written in 2 parts.
"""

import hashlib
import os
import re
import shlex
import subprocess
import sys
import typing
from pathlib import Path

import pytest
import yaml

import m03_03_simulate_new_data as m03_03
import m11_02_generate_months as m11_02
from p1_01_generate_initial_dataset import generate_legitimate_transactions
from p5_04_inference_api import TransactionRequest

PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = Path("data/credit_card_transactions_latest.csv")
BASE_COLUMNS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
              "merchant_category", "card_present", "international", "transaction_count_24h", "is_fraud"]
BASE_CATEGORIES = set(typing.get_args(TransactionRequest.model_fields["merchant_category"].annotation))
# The 4 month files of the original month generator (p7_02) on 2026-10-01 (Python 3.14.7, numpy 2.4.1, pandas 2.3.3)
BASE_MONTH_MD5 = {
    "month_1": "6b71065500c914d54cbc5e51fa2a9112",
    "month_2": "84a269e8c27174d19b80c18e40adbeee",
    "month_3": "d33accaab80d6974" "28656986e01bc842",   # 2 parts: no 12-digit number in this file
    "month_4": "131676d7bd99219b722385aa905fabf1",
}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def rows(path):
    """Data rows in a CSV file (the header line not counted)."""
    return len(Path(path).read_text().splitlines()) - 1


def simulate(*args):
    """Run m03_03 in this process, in the current folder (tmp_path)."""
    return m03_03.main([str(arg) for arg in args])


def seed_in(command):
    """The number after --seed in a command, or None when the command has no --seed."""
    found = re.search(r"--seed (\d+)", command)
    return int(found.group(1)) if found else None


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def base_data(in_tmp):
    """A small the base project data file to append to: 1,000 rows, seed 7."""
    assert simulate("--mode", "refresh", "--rows", 1000, "--seed", 7) == 0
    return in_tmp / DATA_FILE


@pytest.fixture(scope="module")
def manifest():
    for path in (PROJECT / "manifest.yaml", PROJECT.parent / "manifest.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text())
    if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
        pytest.fail("MLOPS_REQUIRE_MODEL=1, but manifest.yaml is missing (not next to ccfd-project/ and not in it)")
    pytest.skip("manifest.yaml not found next to ccfd-project/ or in it")


def run_manifest_command(command, cwd):
    """Run a manifest.yaml command ('python <script> <options>') with this Python, in cwd."""
    words = shlex.split(command)
    assert words[0] == "python"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}     # the base project's p1_01 prints emoji
    result = subprocess.run([sys.executable, str(PROJECT / words[1]), *words[2:]], check=False,
                            cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


# --- the base project's schema ---------------------------------------------------------------------------------------

def test_base_columns_are_the_columns_of_base_code():
    assert m03_03.BASE_COLUMNS == BASE_COLUMNS
    assert list(generate_legitimate_transactions(n=10).columns) == BASE_COLUMNS
    assert len(BASE_CATEGORIES) == 12
    assert set(m11_02.MERCHANTS) == BASE_CATEGORIES


def test_new_data_has_base_columns_12_categories_and_base_ranges():
    df = m03_03.make_rows(2000, 301)
    assert list(df.columns) == BASE_COLUMNS
    assert set(df["merchant_category"]) == BASE_CATEGORIES          # all 12 appear in 2,000 rows
    assert df.notna().all().all()
    assert df["transaction_amount"].between(5, 5000).all()        # p1_01's noise clips amounts to 5..5000
    assert df["avg_transaction_amount"].between(20, 1500).all()
    assert df["transaction_hour"].between(0, 23).all()
    assert set(df["days_since_last_txn"]) <= {0, 1, 2, 3, 4, 5, 6, 7, 14, 30}
    assert df["transaction_count_24h"].between(1, 8).all()
    assert set(df["card_present"]) == {"yes", "no"}
    assert set(df["international"]) == {"yes", "no"}
    assert set(df["is_fraud"]) == {0, 1}
    for record in df.drop(columns="is_fraud").to_dict("records"):
        TransactionRequest(**record)                               # every row is a valid the base project request


@pytest.mark.parametrize("month", m11_02.MONTHS, ids=[m["name"] for m in m11_02.MONTHS])
def test_months_have_base_columns_categories_and_ranges(month):
    df, n_classic, n_stealth = m11_02.make_month(month["amount_inflation"], month["intl_share"],
                                                 month["stealth_frac"], month["seed"])
    assert list(df.columns) == BASE_COLUMNS
    assert set(df["merchant_category"]) <= BASE_CATEGORIES
    assert len(df) == 5000 and df["is_fraud"].sum() == 100 == n_classic + n_stealth   # 2 percent, no flips
    assert df["transaction_amount"].between(5, 8000).all()        # p7_02's light noise clips to 5..8000
    assert df["avg_transaction_amount"].between(20, 1500 * month["amount_inflation"]).all()
    assert df["transaction_hour"].between(0, 23).all()
    assert df["transaction_count_24h"].between(1, 8).all()
    for record in df.drop(columns="is_fraud").to_dict("records"):
        TransactionRequest(**record)


# --- the same seed makes the same file -------------------------------------------------------------------

def test_same_seed_same_md5_other_seed_other_md5(in_tmp):
    simulate("--mode", "refresh", "--rows", 2000, "--seed", 301, "--output", "a.csv")
    simulate("--mode", "refresh", "--rows", 2000, "--seed", 301, "--output", "b.csv")
    simulate("--mode", "refresh", "--rows", 2000, "--seed", 302, "--output", "c.csv")
    assert md5("a.csv") == md5("b.csv")
    assert md5("a.csv") != md5("c.csv")


def test_refresh_with_base_seed_and_size_makes_data_v1(in_tmp, manifest):
    simulate("--mode", "refresh", "--rows", 10000, "--seed", 42, "--output", "v1.csv")
    assert md5("v1.csv") == manifest["data"]["v1"]["md5"]          # m03_03 uses the base project's p1_01 generators exactly


def test_manifest_commands_make_the_manifest_md5s(tmp_path, manifest):
    """The manifest.yaml commands make data v1, v2 (and v3 once recorded) and the gate holdout again, same md5."""
    for name in ("v1", "v2", "v3"):
        version = manifest["data"].get(name)
        if not isinstance(version, dict):
            continue                                                # not recorded yet ('pending')
        run_manifest_command(version["command"], tmp_path)
        assert rows(tmp_path / version["file"]) == version["rows"], name
        assert md5(tmp_path / version["file"]) == version["md5"], name
    data_md5 = md5(tmp_path / DATA_FILE)
    gate = manifest["gate_holdout"]
    run_manifest_command(gate["command"], tmp_path)
    assert gate["file"] == "data/gate_holdout.csv" and gate["s3_key"] == "gate/gate_holdout.csv"
    assert rows(tmp_path / gate["file"]) == gate["rows"] == 2000
    assert md5(tmp_path / gate["file"]) == gate["md5"]
    assert md5(tmp_path / DATA_FILE) == data_md5                   # the gate holdout left the data version alone


def test_months_equal_base_month_files(tmp_path):
    assert [m["seed"] for m in m11_02.MONTHS] == [101, 102, 103, 104]   # the original month generator (p7_02) seeds
    assert m11_02.main(["--data-dir", str(tmp_path)]) == 0
    for name, expected in BASE_MONTH_MD5.items():
        path = tmp_path / f"production_{name}.csv"
        assert rows(path) == 5000
        assert md5(path) == expected, name


def test_no_seed_is_used_twice(manifest):
    """Every data version, the gate holdout and every month has its own seed, read from the code and manifest.yaml."""
    p1_01 = (PROJECT / "p1_01_generate_initial_dataset.py").read_text()
    seeds = {"data v1": int(re.search(r"np\.random\.seed\((\d+)\)", p1_01).group(1))}       # the base project's p1_01
    commands = {f"data {name}": version["command"] for name, version in manifest["data"].items()
                if isinstance(version, dict)}                                              # 'pending' left out
    commands["gate holdout"] = manifest["gate_holdout"]["command"]
    for name, command in commands.items():
        if seed_in(command) is not None:
            seeds[name] = seed_in(command)
    # m03_03's docstring lists every data version, also one that manifest.yaml does not record yet (data v3)
    in_docstring = re.findall(r"^\s+(data v\d|gate holdout)\s+python .*?--seed (\d+)", m03_03.__doc__, re.MULTILINE)
    for name, seed in in_docstring:
        assert seeds.setdefault(name, int(seed)) == int(seed), f"{name}: m03_03's docstring and manifest.yaml differ"
    for month in m11_02.MONTHS:
        seeds[month["name"]] = month["seed"]
    assert {"data v1", "data v2", "data v3", "gate holdout", "month_1", "month_4"} <= set(seeds)
    used = list(seeds.values())
    assert len(set(used)) == len(used), seeds


# --- append and refresh --------------------------------------------------------------------------------

def test_append_adds_rows_at_the_end_and_keeps_old_rows(base_data, capsys):
    old_bytes = base_data.read_bytes()
    capsys.readouterr()
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8) == 0
    assert rows(base_data) == 1500
    assert base_data.read_bytes().startswith(old_bytes)           # the old rows, byte for byte
    out = capsys.readouterr().out
    assert "1,000 + 500 = 1,500 rows (fraud rate" in out
    assert "Next:     dvc add data/credit_card_transactions_latest.csv" in out
    out.encode("ascii")                                            # ASCII output only


def test_append_twice_with_the_same_seed_stops(base_data, capsys):
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8) == 0
    after_first = base_data.read_bytes()
    capsys.readouterr()
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8) == 1
    assert "these 500 rows (seed 8) are already at the end of" in capsys.readouterr().out
    assert base_data.read_bytes() == after_first                  # nothing written
    assert simulate("--mode", "append", "--rows", 500, "--seed", 10) == 0   # a new seed works
    assert rows(base_data) == 2000


def test_refresh_replaces_all_rows(base_data, capsys):
    assert simulate("--mode", "refresh", "--rows", 300, "--seed", 9) == 0
    assert rows(base_data) == 300
    assert "it had 1,000 rows; they are replaced" in capsys.readouterr().out


def test_refresh_repairs_an_empty_or_broken_file(in_tmp, capsys):
    holdout = Path("data/gate_holdout.csv")
    holdout.parent.mkdir()
    holdout.write_bytes(b"")                                       # left by a run that stopped
    assert simulate("--mode", "refresh", "--rows", 100, "--seed", 3, "--output", holdout) == 0
    assert "(it had 0 rows; they are replaced)" in capsys.readouterr().out
    assert rows(holdout) == 100
    holdout.write_bytes(b"\xff\xfe\x00 not a CSV file")
    assert simulate("--mode", "refresh", "--rows", 100, "--seed", 3, "--output", holdout) == 0
    assert "(it was not a readable CSV file; it is replaced)" in capsys.readouterr().out
    assert rows(holdout) == 100


def test_output_never_touches_the_default_file(base_data):
    before = (md5(base_data), base_data.stat().st_mtime_ns)
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8, "--output", "data/other.csv") == 0
    assert simulate("--mode", "refresh", "--rows", 2000, "--seed", 301, "--output", "data/gate_holdout.csv") == 0
    assert (md5(base_data), base_data.stat().st_mtime_ns) == before
    assert rows("data/other.csv") == 1500
    assert rows("data/gate_holdout.csv") == 2000


def test_append_to_another_file_says_what_it_replaces(base_data, capsys):
    Path("data/other.csv").write_text("a,b\n1,2\n3,4\n")
    capsys.readouterr()
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8, "--output", "data/other.csv") == 0
    out = capsys.readouterr().out
    assert "Written:  data/other.csv (it had 2 rows; they are replaced)" in out
    assert "Next:" not in out                                      # not the base project's data file: no dvc add hint
    assert rows("data/other.csv") == 1500


def test_full_path_to_the_data_file_is_the_data_file(base_data, capsys):
    capsys.readouterr()
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8, "--output", base_data.resolve()) == 0
    out = capsys.readouterr().out
    assert "Next:     dvc add data/credit_card_transactions_latest.csv" in out
    assert "replaced" not in out
    assert rows(base_data) == 1500


def test_append_stops_without_base_data_file(in_tmp, capsys):
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8, "--output", "out.csv") == 1
    assert "not found" in capsys.readouterr().out
    assert not Path("out.csv").exists()


def test_append_stops_on_other_columns(base_data, capsys):
    text = base_data.read_text().replace("is_fraud", "label", 1)
    base_data.write_text(text)
    assert simulate("--mode", "append", "--rows", 500, "--seed", 8) == 1
    assert "does not have the base project's 9 columns" in capsys.readouterr().out
    assert base_data.read_text() == text


def test_too_few_rows_is_an_error(in_tmp):
    with pytest.raises(SystemExit) as error:
        simulate("--mode", "refresh", "--rows", 49, "--seed", 1)
    assert error.value.code == 2
    assert not DATA_FILE.exists()


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_out_of_range_is_an_error(in_tmp, seed, capsys):
    with pytest.raises(SystemExit) as error:
        simulate("--mode", "refresh", "--rows", 100, "--seed", seed)
    assert error.value.code == 2
    assert "--seed must be between 0 and 4294967295" in capsys.readouterr().err
    assert not DATA_FILE.exists()


def test_output_folder_is_an_error(in_tmp, capsys):
    Path("data").mkdir()
    with pytest.raises(SystemExit) as error:
        simulate("--mode", "refresh", "--rows", 100, "--seed", 3, "--output", "data")
    assert error.value.code == 2
    assert "--output data is a folder" in capsys.readouterr().err


def test_months_output_is_ascii(tmp_path, capsys):
    m11_02.main(["--data-dir", str(tmp_path)])
    capsys.readouterr().out.encode("ascii")
