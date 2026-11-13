"""
Test helper: package a laptop model as a SageMaker model.tar.gz (used by the tests).

SageMaker wants a model as one model.tar.gz. Our serving image reads 3 files at the root of the unpacked
folder /opt/ml/model:
    model_latest.pkl        the fitted base project Pipeline
    metadata_latest.json    the base project's metadata (model_name, metrics, best params)
    evaluation.json         the base project's results JSON for that model (save_metrics_json in results/): model_info,
                            model_parameters, technical_metrics, business_metrics (net_benefit ...), dataset_info.
                            Every course evaluation.json has this one shape.

A SageMaker training job writes this layout itself (m08_03_train_entrypoint.py). This script makes the same
tar from a model folder on your machine, for example the base project's p5_01 (random search) model: the tests use it
to check the layout and the gate. No section registers its output: every package of the course comes from a SageMaker
training job (MLOps 09; its Rejected package is the data v2 retrain, which MLOps 12 uses for its rollback drill).
The tar is the same, byte for byte, every time it is made from the same 3 files.

Usage (from the ccfd-project folder):
    python tests/package_model_tar.py                                        # p5_01, the weaker model
    python tests/package_model_tar.py --model-dir models/p3_01_baseline --output sm-local/packages/p3_01_baseline.tar.gz
    python tests/package_model_tar.py --output s3://ccfd-mlops-jobs-<random6>/models/p5_01_randomsearch/model.tar.gz
The last line uploads the tar to S3 (an s3:// output is the only case that uses boto3).

Where evaluation.json comes from: --evaluation if given, else the model folder's own evaluation.json (a training
output), else the newest JSON in results/ that base project's save_metrics_json wrote for that model (its
dataset_info.version is the model_name in metadata_latest.json). It must be the base project's results JSON (business_metrics
with net_benefit), and its Net Benefit must equal the one in metadata_latest.json, so a file in another shape or a
results file of another model is never packed by mistake.
boto3 is used only when --output is an s3:// URI. An AWS error is printed with the account ID masked.
"""

import argparse
import gzip
import io
import json
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

# The ccfd-project folder (this file is in its tests/ folder) on sys.path, so "from mlops_utils import config" also
# works when this file runs as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# mlops_utils/config.py uses the standard library only; config.mask() hides a 12-digit account ID
from mlops_utils import config

HERE = Path(__file__).resolve().parents[1]   # the ccfd-project folder
DEFAULT_MODEL_DIR = HERE / "models" / "p5_01_randomsearch"     # the base project's weaker model (p5_01, 37,025 USD)
DEFAULT_RESULTS_DIR = HERE / "results"
MODEL_FILES = ("model_latest.pkl", "metadata_latest.json", "evaluation.json")   # the tar root


def find_evaluation(model_dir: Path, results_dir: Path, model_name: str) -> Path:
    """The evaluation.json source for a model folder: its own file, else the base project's newest results JSON for model_name."""
    own = model_dir / "evaluation.json"
    if own.is_file():
        return own
    found = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(content, dict) and content.get("dataset_info", {}).get("version") == model_name:
            found.append((content.get("model_info", {}).get("timestamp", ""), path))
    if not found:
        raise FileNotFoundError(
            f"No base project results file for {model_name} in {results_dir} (a JSON with dataset_info.version "
            f"'{model_name}'). Run the base project script that made the model, or give --evaluation <file>.")
    return max(found)[1]


def check_same_model(metadata: dict, evaluation: dict, source: Path) -> None:
    """
    evaluation.json must be the base project's results JSON (business_metrics.net_benefit), and its Net Benefit must be the
    one in metadata_latest.json.
    """
    business = evaluation.get("business_metrics") if isinstance(evaluation, dict) else None
    if not isinstance(business, dict) or "net_benefit" not in business:
        raise ValueError(f"{source.name} is not the base project's results JSON: evaluation.json needs business_metrics with "
                         "net_benefit, as the base project's save_metrics_json writes it.")
    expected = metadata.get("metrics", {}).get("net_benefit")
    found = business["net_benefit"]
    if expected is not None and int(float(expected)) != int(float(found)):
        raise ValueError(f"{source.name} has Net Benefit {found}, but metadata_latest.json says {expected}: "
                         "it belongs to another model.")


def tar_bytes(sources: dict[str, Path]) -> bytes:
    """
    A model.tar.gz in memory: each file at the tar root under its file name, in the fixed order.
    The same files give the same bytes every time: fixed time stamps, owner and mode, gzip time 0.
    """
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in MODEL_FILES:
            data = sources[name].read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    packed = io.BytesIO()
    with gzip.GzipFile(fileobj=packed, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return packed.getvalue()


def package(model_dir: Path, output: str, results_dir: Path = DEFAULT_RESULTS_DIR,
            evaluation: Path | None = None, s3_client=None) -> dict:
    """
    Tar model_dir in the model.tar.gz layout to output (a local file path or an s3://bucket/key URI).
    Returns what was written: the output, the size, the model name and its Net Benefit.
    """
    model_dir = Path(model_dir)
    for name in MODEL_FILES[:2]:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"{model_dir} has no {name}: is this a base project model folder?")
    metadata = json.loads((model_dir / "metadata_latest.json").read_text(encoding="utf-8"))
    sources = {
        "model_latest.pkl": model_dir / "model_latest.pkl",
        "metadata_latest.json": model_dir / "metadata_latest.json",
        "evaluation.json": (Path(evaluation) if evaluation
                            else find_evaluation(model_dir, Path(results_dir), metadata.get("model_name"))),
    }
    evaluation_json = json.loads(sources["evaluation.json"].read_text(encoding="utf-8"))
    check_same_model(metadata, evaluation_json, sources["evaluation.json"])
    data = tar_bytes(sources)

    if output.startswith("s3://"):
        bucket, _, key = output[len("s3://"):].partition("/")
        if not bucket or not key or key.endswith("/"):
            raise ValueError(f"{output} is not an s3://bucket/key URI of a file (for example .../model.tar.gz).")
        if s3_client is None:
            # boto3 is imported only for the S3 case
            import boto3

            s3_client = boto3.client("s3", region_name=config.REGION)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "model.tar.gz"
            local.write_bytes(data)
            s3_client.upload_file(str(local), bucket, key)
    else:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    return {
        "output": output,
        "bytes": len(data),
        "files": list(MODEL_FILES),
        "model_name": metadata.get("model_name"),
        "net_benefit": evaluation_json.get("business_metrics", {}).get("net_benefit"),
        "evaluation_source": sources["evaluation.json"].name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a laptop model folder as model.tar.gz (model.tar.gz layout).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="the model folder (default: models/p5_01_randomsearch, the base project's weaker model)")
    parser.add_argument("--output", default=None,
                        help="a local .tar.gz path or s3://bucket/key (default: sm-local/packages/<folder>.tar.gz)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="where the base project's results JSON files are (default: results/)")
    parser.add_argument("--evaluation", type=Path, default=None,
                        help="the JSON file to use as evaluation.json (default: found in --results-dir)")
    args = parser.parse_args(argv)
    output = args.output or str(HERE / "sm-local" / "packages" / f"{args.model_dir.name}.tar.gz")
    try:
        result = package(args.model_dir, output, args.results_dir, args.evaluation)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # For example an S3 error from upload_file: its text can hold the caller's ARN with the account ID,
        # so the traceback and the message are masked; SystemExit(1) then ends the script without
        # printing the unmasked traceback (the same as m13_03)
        print(config.mask(traceback.format_exc()), file=sys.stderr, flush=True)
        print(config.mask(f"ERROR: {type(error).__name__}: {error}"), file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    net_benefit = result["net_benefit"]
    money = f"{net_benefit:,} USD" if isinstance(net_benefit, int) else "not in evaluation.json"
    print(f"Packaged {result['model_name']} (Net Benefit {money}): {', '.join(result['files'])} "
          f"(evaluation.json from {result['evaluation_source']}) -> {result['output']} ({result['bytes']:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
