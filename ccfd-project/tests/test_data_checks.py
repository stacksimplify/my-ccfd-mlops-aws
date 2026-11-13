"""
Tests for m08_02_data_checks.py, the data checks + amount_ratio processing job.

The CLI tests build a fake /opt/ml/processing folder in a temporary folder, the way a processing job fills it:
  input/raw/<one file, no extension>      named like a DVC 3 object (the md5 without its first 2 characters)
and run the script with --processing-dir <that folder>, as "python m08_02_data_checks.py".
The small tests call hard_rules() on made-up rows, so every count is known in advance.

What is proved:
  - data v1 (the base project's p1_01, seed 42) breaks no hard rule, and the fe output is the base project's p4_01
    file (the single file in output/fe)
  - a missing required column, an empty file and the 3 hard rules fail the data checks (H1 a label not 0 or 1, H2 an
    empty cell, H3 a value the serving API would refuse): the report says FAIL with the rows, and output/fe stays empty
  - anything else (another column order, an extra column, a repeated row) passes, and no row is changed or dropped
  - when the job ends anyway (not exactly 1 input file, or the base project's p4_01 cannot compute amount_ratio), the report on
    disk says FAIL with the reason, and output/fe stays empty
  - the limits and the 12 categories come from the base project's own TransactionRequest (imported, not copied)
The data v1 test skips without the laptop data, unless MLOPS_REQUIRE_MODEL=1 (see tests/conftest.py).
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import typing
from pathlib import Path

import pandas as pd
import pytest

import m08_02_data_checks as checks
from utils.feature_engineering_utils import apply_feature_engineering

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "m08_02_data_checks.py"
DATA_V1 = PROJECT / "data" / "credit_card_transactions_latest.csv"
DATA_V1_MD5 = "65ce973256292d47137e8ddb89840528"      # manifest.yaml data.v1.md5 (the base project's p1_01, seed 42)
BASE_FE_FILE = "credit_card_transactions_fe1_amount_ratio.csv"
BASE_COLUMNS = ["transaction_amount", "transaction_hour", "days_since_last_txn", "avg_transaction_amount",
              "merchant_category", "card_present", "international", "transaction_count_24h", "is_fraud"]
# One transaction that passes every check: count 1 (only this purchase) and days 4 agree
VALID = {"transaction_amount": 82.04, "transaction_hour": 11, "days_since_last_txn": 4,
         "avg_transaction_amount": 227.88, "merchant_category": "pharmacy", "card_present": "yes",
         "international": "no", "transaction_count_24h": 1, "is_fraud": 0}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def need_data_v1():
    if not DATA_V1.exists() or md5(DATA_V1) != DATA_V1_MD5:
        message = "needs data v1 in data/credit_card_transactions_latest.csv: run the base project's p1_01 first"
        if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
            pytest.fail(f"MLOPS_REQUIRE_MODEL=1, but this test {message}")
        pytest.skip(message)


def rows(*changes: dict) -> list[dict]:
    """Made-up transactions: VALID with the given changes, one row per dict."""
    return [{**VALID, **change} for change in changes]


def processing_dir(root: Path, content: bytes | Path, name: str | None = None) -> Path:
    """A processing folder with one input file (bytes, or a copy of a file), named like a DVC object."""
    data = content.read_bytes() if isinstance(content, Path) else content
    folder = root / "processing"
    (folder / "input" / "raw").mkdir(parents=True)
    (folder / "input" / "raw" / (name or hashlib.md5(data).hexdigest()[2:])).write_bytes(data)
    return folder


def csv_bytes(records: list[dict], columns: list[str] | None = None) -> bytes:
    return pd.DataFrame(records, columns=columns).to_csv(index=False).encode()


def run(folder: Path, *extra: str):
    """The script as its own Python process, exactly as the processing job starts it."""
    return subprocess.run([sys.executable, str(SCRIPT), "--processing-dir", str(folder), *extra], cwd=PROJECT,
                          capture_output=True, text=True, check=False)


def call(folder: Path, capsys, *extra: str) -> tuple[int, str]:
    """The same main() inside this process (faster): the exit code and what it printed."""
    code = checks.main(["--processing-dir", str(folder), *extra])
    return code, capsys.readouterr().out


def report_of(folder: Path) -> dict:
    return json.loads((folder / "output" / "report" / "data_checks_report.json").read_text())


def files_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


# ---------------------------------------------------------------------------
# 1. Data v1: no hard rule broken, and the base project's FE file
# ---------------------------------------------------------------------------
def test_data_v1_passes_and_gives_the_base_fe_file(tmp_path):
    need_data_v1()
    folder = processing_dir(tmp_path, DATA_V1)
    input_file = next((folder / "input" / "raw").iterdir())
    assert "." not in input_file.name                               # a DVC object name, no extension
    result = run(folder)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    report = report_of(folder)
    assert report["status"] == "PASS" and report["blocking_problems"] == []
    assert report["checked_at"].endswith("+00:00")                  # UTC, the same on the laptop and in a job
    assert report["input"] == {"file": input_file.name, "bytes": DATA_V1.stat().st_size, "md5": DATA_V1_MD5}
    assert report["rows"] == 10000
    assert report["hard_rules"] == {"H1": {"check": "labels", "rows": 0, "first_rows": []},
                                    "H2": {"check": "nulls", "rows": 0, "first_rows": []},
                                    "H3": {"check": "ranges", "rows": 0, "first_rows": []}}
    assert list(report) == ["script", "checked_at", "status", "blocking_problems", "input", "rows", "hard_rules", "fe"]
    # The outputs: channel fe holds exactly the base project's FE file, the report has its own folder
    assert files_in(folder / "output" / "fe") == [BASE_FE_FILE]
    assert files_in(folder / "output" / "report") == ["data_checks_report.json"]
    fe_out = folder / "output" / "fe" / BASE_FE_FILE
    assert report["fe"]["md5"] == md5(fe_out) and report["fe"]["rows"] == 10000
    # The base project's p4_01 made it: the same frame as the base project's apply_feature_engineering on data v1 (m08_02 has no FE code)
    pd.testing.assert_frame_equal(pd.read_csv(fe_out), apply_feature_engineering(pd.read_csv(DATA_V1)))
    base_fe = PROJECT / "data" / BASE_FE_FILE                           # the base project's own p4_01 output on data v1 (Step-01)
    if base_fe.exists():
        assert md5(fe_out) == md5(base_fe)
    assert md5(input_file) == DATA_V1_MD5                            # the input is never changed
    assert str(tmp_path) not in json.dumps(report)                   # the report holds file names, not paths
    # The console: rows, the hard rules, p4_01 between its 2 === lines, the fe line, PASS
    lines = [line for line in result.stdout.splitlines() if line.startswith("[m08_02]")]
    assert lines[:2] == ["[m08_02] 10,000 rows", "[m08_02] Hard rules: H1 0, H2 0, H3 0"]
    assert lines[2] == "[m08_02] === The base project p4_01_feature_engineering_fe1.py"
    assert lines[4].startswith(f"[m08_02] fe: {BASE_FE_FILE} (10,000 rows, md5 {report['fe']['md5']}) -> ")
    assert lines[5:] == ["[m08_02] PASS"]


def test_fe_output_is_what_the_training_entrypoint_reads(tmp_path):
    """the training job's channel fe is this folder; m08_03 takes 'the single file in the channel'."""
    import m08_03_train_entrypoint as train

    folder = processing_dir(tmp_path, csv_bytes(rows({}, {"is_fraud": 1, "transaction_count_24h": 3,
                                                          "days_since_last_txn": 0})))
    assert checks.main(["--processing-dir", str(folder)]) == 0
    fe_file = train.single_file(folder / "output" / "fe")
    assert fe_file.name == BASE_FE_FILE
    fe = pd.read_csv(fe_file)
    assert list(fe.columns) == [*BASE_COLUMNS, "amount_ratio"]
    assert fe["amount_ratio"].tolist() == pytest.approx([82.04 / 227.88, 82.04 / 227.88])


# ---------------------------------------------------------------------------
# 2. What fails the job: a missing column, an empty file, the 3 hard rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("column", ["is_fraud", "merchant_category"])
def test_missing_required_column_fails(tmp_path, capsys, column):
    columns = [name for name in BASE_COLUMNS if name != column]
    folder = processing_dir(tmp_path, csv_bytes(rows({}), columns))
    code, out = call(folder, capsys)
    assert code == 1
    assert f"FAILED: Required column(s) missing: {column}." in out
    report = report_of(folder)
    assert report["status"] == "FAIL" and column in report["blocking_problems"][0] and report["fe"] is None
    assert files_in(folder / "output" / "fe") == []                  # nothing for the training job


@pytest.mark.parametrize("content", [b"", b"\n\n", ",".join(BASE_COLUMNS).encode() + b"\n"],
                         ids=["no-bytes", "blank-lines", "header-only"])
def test_empty_file_fails(tmp_path, capsys, content):
    folder = processing_dir(tmp_path, content, name="ce973256292d47137e8ddb89840528")
    code, out = call(folder, capsys)
    assert code == 1
    assert "is empty" in out
    assert report_of(folder)["status"] == "FAIL"
    assert files_in(folder / "output" / "fe") == []


def test_input_folder_must_hold_exactly_one_file(tmp_path, capsys):
    folder = processing_dir(tmp_path, csv_bytes(rows({})))
    (folder / "input" / "raw" / "second.csv").write_bytes(csv_bytes(rows({})))
    code, out = call(folder, capsys)
    assert code == 1
    assert "must hold exactly 1 file, found 2" in out
    report = report_of(folder)                                       # the report says why the job ended
    assert report["status"] == "FAIL" and "found 2" in report["blocking_problems"][0] and report["input"] is None
    assert files_in(folder / "output" / "fe") == []


def test_a_file_that_is_not_csv_names_the_bad_line(tmp_path, capsys):
    """An unreadable file counts as 'required columns missing'; pandas' own text names the bad line."""
    good = csv_bytes(rows({})).decode()
    content = (good + good.splitlines()[1] + ",EXTRA\n").encode()     # line 3 has a 10th field
    folder = processing_dir(tmp_path, content, name="abc123")
    code, _ = call(folder, capsys)
    assert code == 1
    reason = report_of(folder)["blocking_problems"][0]
    assert "abc123 cannot be read as CSV (ParserError" in reason and "Expected 9 fields in line 3, saw 10" in reason
    assert files_in(folder / "output" / "fe") == []


def test_hard_rule_h1_wrong_label_fails(tmp_path, capsys):
    """H1: a label that is not 0 or 1 (MLOps 08 "See the door close": 3 labels set to 2). No FE file, so no training."""
    raw = csv_bytes(rows({"is_fraud": 2}, {"is_fraud": 2}, {"is_fraud": 2}, {}, {"is_fraud": 1}))
    folder = processing_dir(tmp_path, raw)
    code, out = call(folder, capsys)
    assert code == 1
    assert out.splitlines() == [
        "[m08_02] 5 rows",
        "[m08_02] Hard rules: H1 3, H2 0, H3 0",
        "[m08_02] FAILED: hard rule H1 labels: 3 row(s) where is_fraud is not 0 or 1 (first rows: 0, 1, 2)"]
    report = report_of(folder)
    assert report["status"] == "FAIL" and report["fe"] is None
    assert report["blocking_problems"] == ["hard rule H1 labels: 3 row(s) where is_fraud is not 0 or 1 (first rows: 0, 1, 2)"]
    assert report["hard_rules"]["H1"] == {"check": "labels", "rows": 3, "first_rows": [0, 1, 2]}
    assert files_in(folder / "output" / "fe") == []                  # the door is closed: nothing for the training job
    assert md5(next((folder / "input" / "raw").iterdir())) == hashlib.md5(raw).hexdigest()   # the input is never changed


def test_hard_rule_h2_empty_cell_fails(tmp_path, capsys):
    folder = processing_dir(tmp_path, csv_bytes(rows({}, {"avg_transaction_amount": None}, {"is_fraud": 1})))
    code, out = call(folder, capsys)
    assert code == 1
    assert "FAILED: hard rule H2 nulls: 1 row(s) with an empty cell (first rows: 1)" in out
    report = report_of(folder)
    assert report["status"] == "FAIL" and report["hard_rules"]["H2"]["rows"] == 1
    assert report["hard_rules"]["H3"]["rows"] == 0                   # an empty cell counts once, under H2
    assert files_in(folder / "output" / "fe") == []


@pytest.mark.parametrize("change", [{"transaction_hour": 25}, {"transaction_amount": -5.0},
                                    {"transaction_amount": math.inf}, {"merchant_category": "Grocery"},
                                    {"card_present": "maybe"}],
                         ids=["hour-25", "negative-amount", "inf", "unknown-category", "not-yes-or-no"])
def test_hard_rule_h3_value_the_api_refuses_fails(tmp_path, capsys, change):
    folder = processing_dir(tmp_path, csv_bytes(rows({}, change, {"is_fraud": 1})))
    code, out = call(folder, capsys)
    assert code == 1
    assert "FAILED: hard rule H3 ranges: 1 row(s) with a value the serving API would refuse (first rows: 1)" in out
    report = report_of(folder)
    assert report["status"] == "FAIL" and report["hard_rules"]["H3"]["rows"] == 1
    assert files_in(folder / "output" / "fe") == []


def test_text_in_an_amount_is_hard_rule_h3_before_p4_01(tmp_path):
    """Text in transaction_amount: the API refuses it (H3), so the job fails before the base project's p4_01 runs."""
    records = frame({}, {}).astype({"transaction_amount": object})
    records.loc[1, "transaction_amount"] = "abc"
    folder = processing_dir(tmp_path, records.to_csv(index=False).encode())
    result = run(folder)
    assert result.returncode == 1
    assert "FAILED: hard rule H3 ranges: 1 row(s)" in result.stdout
    assert "=== The base project p4_01" not in result.stdout             # p4_01 never started
    report = report_of(folder)
    assert report["status"] == "FAIL" and report["fe"] is None
    assert report["hard_rules"]["H3"] == {"check": "ranges", "rows": 1, "first_rows": [1]}
    assert files_in(folder / "output" / "fe") == []


def test_every_broken_hard_rule_is_named(tmp_path, capsys):
    folder = processing_dir(tmp_path, csv_bytes(rows({"is_fraud": 2}, {"avg_transaction_amount": None},
                                                     {"transaction_hour": 25})))
    code, out = call(folder, capsys)
    assert code == 1
    assert "[m08_02] Hard rules: H1 1, H2 1, H3 1" in out
    problems = report_of(folder)["blocking_problems"]
    assert [problem.split(":")[0] for problem in problems] == ["hard rule H1 labels", "hard rule H2 nulls",
                                                               "hard rule H3 ranges"]


def test_p4_01_failure_leaves_a_fail_report_and_no_fe_file(tmp_path, capsys, monkeypatch):
    """Data that passes every rule, but the base project's p4_01 stops (simulated): the job exits 1 and the report on
    disk says FAIL with that reason (never PASS)."""
    def p4_01_stops(raw, fe_dir, code_dir=None):
        fe_dir.mkdir(parents=True, exist_ok=True)
        (fe_dir / BASE_FE_FILE).write_text("half a file")               # a half-written output is removed too
        raise checks.CheckError("it exited with code 1; its own error is in the log above")

    monkeypatch.setattr(checks, "run_p4_01", p4_01_stops)
    folder = processing_dir(tmp_path, csv_bytes(rows({}, {"is_fraud": 1})))
    code, out = call(folder, capsys)
    assert code == 1
    assert "FAILED: The base project's p4_01_feature_engineering_fe1.py could not compute amount_ratio" in out
    report = report_of(folder)
    assert report["status"] == "FAIL" and report["fe"] is None
    assert len(report["blocking_problems"]) == 1
    assert "could not compute amount_ratio" in report["blocking_problems"][0]
    assert files_in(folder / "output" / "fe") == []


def test_input_argument_takes_a_file(tmp_path, capsys):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(csv_bytes(rows({}, {"is_fraud": 1})))
    folder = tmp_path / "processing"
    code, out = call(folder, capsys, "--input", str(raw))
    assert code == 0, out[-2000:]
    assert report_of(folder)["input"]["file"] == "raw.csv"
    assert files_in(folder / "output" / "fe") == [BASE_FE_FILE]


# ---------------------------------------------------------------------------
# 3. Anything that breaks no hard rule passes, and no row changes
# ---------------------------------------------------------------------------
def test_other_problems_pass_and_no_row_changes(tmp_path):
    records = rows(
        {},                                                            # 0 clean, legit
        {"is_fraud": 1},                                               # 1 clean, fraud
        {},                                                            # 2 a repeat of row 0
        {"days_since_last_txn": 3, "transaction_count_24h": 4, "is_fraud": 1},   # 3 days and count disagree
    )
    order = ["is_fraud", *BASE_COLUMNS[:-1]]                             # is_fraud moved to the front
    frame = pd.DataFrame(records)[order].assign(note="x")              # plus an extra column
    raw = frame.to_csv(index=False).encode()
    folder = processing_dir(tmp_path, raw)
    result = run(folder)
    assert result.returncode == 0, result.stdout[-3000:]
    report = report_of(folder)
    assert report["status"] == "PASS" and report["rows"] == 4 and report["blocking_problems"] == []
    assert {rule: value["rows"] for rule, value in report["hard_rules"].items()} == {"H1": 0, "H2": 0, "H3": 0}
    # The fe file has every input row, and the input columns hold exactly the input values
    fe = pd.read_csv(folder / "output" / "fe" / BASE_FE_FILE)
    original = pd.read_csv(next((folder / "input" / "raw").iterdir()))
    pd.testing.assert_frame_equal(fe.drop(columns=["amount_ratio"]), original)
    assert md5(next((folder / "input" / "raw").iterdir())) == hashlib.md5(raw).hexdigest()


def frame(*changes: dict) -> pd.DataFrame:
    return pd.DataFrame(rows(*changes))[BASE_COLUMNS]


def counts(df: pd.DataFrame) -> dict:
    return {rule: value["rows"] for rule, value in checks.hard_rules(df).items()}


def test_a_clean_frame_breaks_no_hard_rule():
    assert counts(frame({}, {"is_fraud": 1, "transaction_count_24h": 2, "days_since_last_txn": 0})) == {
        "H1": 0, "H2": 0, "H3": 0}


def test_an_empty_label_is_h1_and_h2():
    assert counts(frame({"is_fraud": None}, {})) == {"H1": 1, "H2": 1, "H3": 0}


def test_an_empty_category_is_h2_not_h3():
    assert counts(frame({"merchant_category": None}, {})) == {"H1": 0, "H2": 1, "H3": 0}


def test_text_in_a_number_column_is_h3():
    records = frame({}, {}).astype({"transaction_hour": object})
    records.loc[1, "transaction_hour"] = "late"
    rules = checks.hard_rules(records)
    assert rules["H3"] == {"check": "ranges", "rows": 1, "first_rows": [1]}


def test_only_the_first_3_rows_are_kept():
    rules = checks.hard_rules(frame(*[{"is_fraud": 5}] * 4))
    assert rules["H1"] == {"check": "labels", "rows": 4, "first_rows": [0, 1, 2]}


# ---------------------------------------------------------------------------
# 4. The rules come from the base project's TransactionRequest, never a copy
# ---------------------------------------------------------------------------
def test_limits_and_categories_come_from_base_transaction_request():
    import p5_04_inference_api as base

    assert checks.TransactionRequest is base.TransactionRequest
    assert checks.REQUIRED == BASE_COLUMNS                                # the base project's 8 fields in the same order + is_fraud
    categories = typing.get_args(base.TransactionRequest.model_fields["merchant_category"].annotation)
    assert len(categories) == 12
    for category in categories:                                           # H3 takes each of the 12, and nothing else
        assert counts(frame({"merchant_category": category}))["H3"] == 0
    assert counts(frame({"merchant_category": "Grocery"}))["H3"] == 1
    source = SCRIPT.read_text()
    assert "'grocery'" not in source and '"grocery"' not in source       # the 12 categories are not copied


def test_data_v1_md5_is_the_manifest_value():
    for path in (PROJECT / "manifest.yaml", PROJECT.parent / "manifest.yaml"):
        if path.is_file():
            import yaml

            assert yaml.safe_load(path.read_text())["data"]["v1"]["md5"] == DATA_V1_MD5
            return
    pytest.skip("manifest.yaml not found next to ccfd-project/ or in it")
