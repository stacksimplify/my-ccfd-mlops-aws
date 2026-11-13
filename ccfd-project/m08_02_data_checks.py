"""
MLOps 08: the data checks, then the base project's amount_ratio, as one SageMaker processing job.

The job runs in the training image (one jobs image) with
    ContainerEntrypoint ["python3", "/opt/program/m08_02_data_checks.py"]

What the job gives the container (the job request sets these folders, mlops_utils/jobs.py):
  /opt/ml/processing/input/raw/<one file>     the raw data with the base project's 9 columns (input "raw", MLOps 08: "the raw
                                              channel"). Any file name works, also a DVC object (its md5, no extension).
What this file writes:
  /opt/ml/processing/output/report/data_checks_report.json
                                              PASS or FAIL, the reason, and the row count of each hard rule
  /opt/ml/processing/output/fe/credit_card_transactions_fe1_amount_ratio.csv
                                              The base project's FE1 file, made by the base project's p4_01 (unchanged). The training job reads it
                                              as its channel fe ("the single file in the channel").
The report has its own output folder, so the fe folder always holds exactly one file.

The checks never change a row: no row is changed, dropped or added, and the base project's p4_01 reads the input file
byte for byte. The job fails (exit code 1, no feature file, so no training) when:
  - a required column is missing (the base project's 9 columns: the 8 fields of p5_04's TransactionRequest and is_fraud; a file
    that cannot be read as CSV counts here, since none of its columns can be found), or
  - the file is empty (no bytes, only blank lines, or a header with no rows), or
  - one of the 3 HARD RULES is broken, in any row:
      H1 labels  is_fraud is 0 or 1 (a wrong label teaches the model the wrong answer)
      H2 nulls   no empty cell (the base project's training cannot use an empty cell)
      H3 ranges  every row passes the base project's TransactionRequest, imported from p5_04 (never copied), and every
                 number is finite (a value the serving API would refuse: amount > 0, hour 0 to 23, one of the 12
                 merchant categories, yes or no ...)
Two more cases end the job, because nothing can be checked or computed:
  - the input is not exactly 1 file (no file, or 2 files), and
  - The base project's p4_01 cannot compute amount_ratio. The base project's code is never changed and the checks never
    change rows, so the job cannot go on.
Every failure writes the report with status FAIL and the reason in blocking_problems, and output/fe stays empty. In a
SageMaker job the reason also goes to /opt/ml/output/message (DescribeProcessingJob shows it as ExitMessage).

On the laptop (or in a test), point it at a folder in the processing layout:
    python m08_02_data_checks.py --processing-dir sm-local/processing/data-checks
or give the input file itself (the outputs still go under --processing-dir):
    python m08_02_data_checks.py --processing-dir sm-local/processing/data-checks \\
        --input data/credit_card_transactions_latest.csv
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

# The base project's request model, unchanged: its 8 fields, their order, limits and the 12 categories
from p5_04_inference_api import TransactionRequest

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# What the checks expect, all read from the base project's TransactionRequest
# ---------------------------------------------------------------------------
FIELDS = list(TransactionRequest.model_fields)            # the base project's 8 input fields, in the same column order
TARGET = "is_fraud"                                        # the base project's label column (the 9th column)
REQUIRED = [*FIELDS, TARGET]                               # a missing one fails the job
NUMBER_FIELDS = [name for name, field in TransactionRequest.model_fields.items() if field.annotation in (int, float)]

# The base project's p4_01 file names (the script reads and writes exactly these, relative to its folder)
P4_01_SCRIPT = "p4_01_feature_engineering_fe1.py"
RAW_FILE = "data/credit_card_transactions_latest.csv"
FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"
REPORT_NAME = "data_checks_report.json"

PROCESSING_DIR = Path("/opt/ml/processing")
INPUT_NAME = "raw"                                         # /opt/ml/processing/input/raw
EXAMPLE_ROWS = 3                                           # row numbers kept per rule (0 = the first data row)

# The 3 hard rules: (id, its name, the words of the FAILED line)
HARD_RULES = (
    ("H1", "labels", "row(s) where is_fraud is not 0 or 1"),
    ("H2", "nulls", "row(s) with an empty cell"),
    ("H3", "ranges", "row(s) with a value the serving API would refuse"),
)


class CheckError(Exception):
    """A problem that stops the job: the message is the job's failure reason."""


class BlockingProblem(CheckError):
    """A data problem that fails the job: a missing required column, an empty file or a broken hard rule."""


def say(message: str) -> None:
    print(f"[m08_02] {message}", flush=True)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. Read the input
# ---------------------------------------------------------------------------
def single_file(source: Path) -> Path:
    """The input: a file, or the one file in a folder whatever its name. Hidden files are skipped."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise CheckError(f"No input file or folder {source}.")
    files = sorted(path for path in source.rglob("*") if path.is_file() and not path.name.startswith("."))
    if len(files) != 1:
        names = ", ".join(str(path.relative_to(source)) for path in files[:5]) or "none"
        raise CheckError(f"Input {source.name} must hold exactly 1 file, found {len(files)} ({names}).")
    return files[0]


def read_data(path: Path) -> pd.DataFrame:
    """Read the file the way the base project's p4_01 does (pd.read_csv). An empty file or a missing column stops the job."""
    if path.stat().st_size == 0:
        raise BlockingProblem(f"The input file {path.name} is empty (no bytes).")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise BlockingProblem(f"The input file {path.name} is empty (no columns, only blank lines).") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        # pandas' own text names the line, for example "Expected 9 fields in line 3, saw 10"
        reason = " ".join(str(error).split())
        raise BlockingProblem(f"The input file {path.name} cannot be read as CSV ({type(error).__name__}: {reason}), "
                              "so the required columns cannot be found.") from error
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise BlockingProblem(f"Required column(s) missing: {', '.join(missing)}. "
                              f"The base project's data has these 9 columns: {', '.join(REQUIRED)}.")
    if df.empty:
        raise BlockingProblem(f"The input file {path.name} is empty (a header with no rows).")
    return df


# ---------------------------------------------------------------------------
# 2. The 3 hard rules (none of them changes the data)
# ---------------------------------------------------------------------------
def api_refuses(df: pd.DataFrame) -> np.ndarray:
    """
    H3: the rows the base project's TransactionRequest (the API's own check) refuses. An empty cell is left to H2.
    A number that is not finite (inf) passes the base project's float fields, so it is flagged here as well.
    """
    bad = np.zeros(len(df), dtype=bool)
    for row, record in enumerate(df[FIELDS].to_dict("records")):
        try:
            TransactionRequest.model_validate(record)
        except ValidationError as error:
            bad[row] = any(not pd.isna(record[problem["loc"][0]]) for problem in error.errors())
    for name in NUMBER_FIELDS:
        bad |= np.isinf(pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float))
    return bad


def hard_rules(df: pd.DataFrame) -> dict:
    """Each hard rule's count of rows that break it, and the first of those rows (0 rows = the rule holds)."""
    flags = {
        "H1": ~pd.to_numeric(df[TARGET], errors="coerce").isin([0, 1]).to_numpy(),
        "H2": df.isna().any(axis=1).to_numpy(),
        "H3": api_refuses(df),
    }
    return {rule: {"check": check, "rows": int(flags[rule].sum()),
                   "first_rows": [int(index) for index in np.flatnonzero(flags[rule])[:EXAMPLE_ROWS]]}
            for rule, check, _ in HARD_RULES}


def hard_rule_problems(rules: dict) -> list[str]:
    """One FAILED reason per broken hard rule, for example
    'hard rule H1 labels: 3 row(s) where is_fraud is not 0 or 1 (first rows: 0, 1, 2)'."""
    return [f"hard rule {rule} {check}: {rules[rule]['rows']:,} {words} "
            f"(first rows: {', '.join(str(row) for row in rules[rule]['first_rows'])})"
            for rule, check, words in HARD_RULES if rules[rule]["rows"]]


# ---------------------------------------------------------------------------
# 3. The base project's p4_01, unchanged, in a work copy
# ---------------------------------------------------------------------------
def run_p4_01(raw: Path, fe_dir: Path, code_dir: Path = HERE) -> dict:
    """Run the base project's p4_01 on the input file (copied byte for byte) and put its FE1 file into the fe output folder."""
    with tempfile.TemporaryDirectory(prefix="base-fe-") as tmp:
        work = Path(tmp)
        shutil.copytree(code_dir / "utils", work / "utils", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(code_dir / P4_01_SCRIPT, work / P4_01_SCRIPT)
        (work / "data").mkdir()
        shutil.copy2(raw, work / RAW_FILE)
        say(f"=== The base project {P4_01_SCRIPT}")
        start = time.time()
        result = subprocess.run([sys.executable, P4_01_SCRIPT], cwd=work, check=False)
        seconds = time.time() - start
        if result.returncode != 0:
            raise CheckError(f"it exited with code {result.returncode}; its own error is in the log above")
        say(f"=== The base project {P4_01_SCRIPT} done in {seconds:.1f} s")
        fe_dir.mkdir(parents=True, exist_ok=True)
        target = fe_dir / Path(FE_FILE).name
        shutil.copy2(work / FE_FILE, target)
    fe = pd.read_csv(target)
    return {"file": target.name, "rows": len(fe), "columns": list(fe.columns), "md5": md5(target),
            "p4_01_seconds": round(seconds, 1)}


# ---------------------------------------------------------------------------
# 4. The job
# ---------------------------------------------------------------------------
def write_report(report: dict, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def fail_report(report: dict, report_dir: Path, reason: str | list[str]) -> None:
    """Write the report with status FAIL and the reason(s), so the report on disk always matches the job's result.
    In a SageMaker job (the default folder) the reason also goes to /opt/ml/output/message, the job's ExitMessage."""
    report["status"] = "FAIL"
    report["blocking_problems"] = [reason] if isinstance(reason, str) else list(reason)
    report["fe"] = None
    write_report(report, report_dir)
    if report_dir.parents[1] == PROCESSING_DIR:
        message = PROCESSING_DIR.parent / "output" / "message"
        try:
            message.parent.mkdir(parents=True, exist_ok=True)
            text = "; ".join(report["blocking_problems"])
            message.write_text(text.encode("ascii", "replace").decode()[:1000], encoding="utf-8")
        except OSError:
            pass                                                       # the log and the report still hold the reason


def check_and_engineer(processing_dir: Path, input_path: Path | None = None) -> dict:
    """The whole job: read, check, report, then the base project's p4_01. Raises CheckError when the job must fail."""
    report_dir = processing_dir / "output" / "report"
    fe_dir = processing_dir / "output" / "fe"
    report = {
        "script": Path(__file__).name,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),     # UTC (+00:00) on the laptop and in a job
        "status": "FAIL",
        "blocking_problems": [],
        "input": None,
    }
    try:
        raw = single_file(input_path or processing_dir / "input" / INPUT_NAME)
        report["input"] = {"file": raw.name, "bytes": raw.stat().st_size, "md5": md5(raw)}
        df = read_data(raw)
    except CheckError as error:
        fail_report(report, report_dir, str(error))
        raise

    report["rows"] = len(df)
    report["hard_rules"] = hard_rules(df)
    say(f"{report['rows']:,} rows")
    say("Hard rules: " + ", ".join(f"{rule} {value['rows']:,}" for rule, value in report["hard_rules"].items()))
    broken = hard_rule_problems(report["hard_rules"])
    if broken:
        # The door closes: no feature file, so the training job (or the pipeline's next step) never starts
        (fe_dir / Path(FE_FILE).name).unlink(missing_ok=True)
        fail_report(report, report_dir, broken)
        raise BlockingProblem("; ".join(broken))
    try:
        report["fe"] = run_p4_01(raw, fe_dir)
    except Exception as error:
        # PASS is written only after p4_01 made the FE file; a failure here leaves a FAIL report and no fe file
        (fe_dir / Path(FE_FILE).name).unlink(missing_ok=True)
        cause = str(error) if isinstance(error, CheckError) else f"{type(error).__name__}: {error}"
        reason = (f"The base project's {P4_01_SCRIPT} could not compute amount_ratio on this data, so there is no FE file for the "
                  f"training job ({cause}).")
        fail_report(report, report_dir, reason)
        if isinstance(error, CheckError):
            raise CheckError(reason) from error
        raise
    report["status"] = "PASS"
    write_report(report, report_dir)
    say(f"fe: {report['fe']['file']} ({report['fe']['rows']:,} rows, md5 {report['fe']['md5']}) -> {fe_dir}")
    say("PASS")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data checks + the base project's amount_ratio (processing job).")
    parser.add_argument("--processing-dir", type=Path, default=PROCESSING_DIR,
                        help="the processing folder (default /opt/ml/processing; on the laptop for example "
                             "sm-local/processing/data-checks)")
    parser.add_argument("--input", type=Path, default=None,
                        help="the raw data file, or a folder with exactly 1 file "
                             f"(default: <processing-dir>/input/{INPUT_NAME})")
    args = parser.parse_args(argv)
    try:
        check_and_engineer(args.processing_dir, args.input)
    except CheckError as error:
        say(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
