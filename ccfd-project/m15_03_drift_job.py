"""
MLOps 15: our drift job (a processing job on ml.t3.medium in the training image; no Model Monitor).

What the job gives the container:
  /opt/ml/processing/input/reference/<one csv>   the reference: the original data (data v1)
  /opt/ml/processing/input/capture/...            the data capture files of ccfd-prod (JSON Lines, any depth)
  /opt/ml/processing/input/labels/*.csv           the late labels (inference_id,is_fraud), only with --with-labels
What it writes:
  /opt/ml/processing/output/report/drift_report.html and drift_summary.json
With --publish-metrics: MaxPSI, NetBenefit and NetBenefitDropPct to CloudWatch namespace MLOps/CCFD, no dimensions, one
value per month in month order; they feed the alarms ccfd-maxpsi-high (>= 0.10) and ccfd-netbenefit-drop
(>= 25 percent below month_1).

Locally:  python m15_03_drift_job.py --processing-dir sm-local/processing/drift --with-labels
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mlops_utils import monitoring
from utils.data_utils import get_feature_columns
from utils.feature_engineering_utils import apply_feature_engineering

PROCESSING_DIR = Path("/opt/ml/processing")
NAMESPACE = "MLOps/CCFD"


def summarize(processing_dir: Path, with_labels: bool) -> dict:
    inputs = processing_dir / "input"
    reference = apply_feature_engineering(pd.read_csv(next((inputs / "reference").glob("*.csv"))))
    categorical, numerical = get_feature_columns(reference)
    texts = [p.read_text() for p in sorted((inputs / "capture").rglob("*.jsonl"))]
    captured = monitoring.read_capture_lines(texts)
    if captured.empty:
        raise SystemExit("no capture records found under input/capture (wait about 5 minutes after the replay)")
    captured["month"] = captured["inference_id"].map(monitoring.month_of)
    months = {}
    for month, rows in sorted(captured[captured["month"] != ""].groupby("month")):
        current = apply_feature_engineering(rows[monitoring.FIELDS].copy())
        report = monitoring.drift_report(reference, current, categorical, numerical)
        months[month] = {"rows": len(rows), "max_psi": report[0]["psi"], "worst_feature": report[0]["feature"],
                         "band": report[0]["band"], "features": report}
    if with_labels:
        labels = pd.concat([pd.read_csv(p) for p in sorted((inputs / "labels").glob("*.csv"))])
        by_month = monitoring.net_benefit_by_month(monitoring.join_ground_truth(captured, labels))
        baseline = by_month.get("month_1")
        for month, net_benefit in by_month.items():
            months.setdefault(month, {})["net_benefit"] = net_benefit
            months[month]["net_benefit_drop_pct"] = monitoring.drop_pct(baseline, net_benefit) if baseline else 0.0
    return {"reference_rows": len(reference), "captured_rows": len(captured), "months": months}


def html(summary: dict) -> str:
    rows = "".join(
        f"<tr><td>{m}</td><td>{v.get('rows', '')}</td><td>{v.get('max_psi', '')}</td><td>{v.get('worst_feature', '')}"
        f"</td><td>{v.get('band', '')}</td><td>{v.get('net_benefit', '')}</td><td>{v.get('net_benefit_drop_pct', '')}"
        f"</td></tr>" for m, v in summary["months"].items())
    return ("<html><body><h1>ccfd-prod drift report</h1><table border='1'><tr><th>month</th><th>rows</th><th>max PSI"
            "</th><th>worst feature</th><th>band</th><th>Net Benefit (USD)</th><th>drop vs month_1 (%)</th></tr>"
            f"{rows}</table></body></html>")


def publish(summary: dict, cloudwatch=None) -> list[dict]:
    """One put_metric_data per month, in month order (the alarms read Maximum over 5 minutes)."""
    import boto3

    cloudwatch = cloudwatch or boto3.client("cloudwatch", region_name="us-east-1")
    sent = []
    for month, values in summary["months"].items():
        data = [{"MetricName": "MaxPSI", "Value": float(values["max_psi"]), "Unit": "None"}]
        if "net_benefit" in values:
            data += [{"MetricName": "NetBenefit", "Value": float(values["net_benefit"]), "Unit": "None"},
                     {"MetricName": "NetBenefitDropPct", "Value": float(values["net_benefit_drop_pct"]), "Unit": "None"}]
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=data)
        sent.append({"month": month, **{d["MetricName"]: d["Value"] for d in data}})
    return sent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Drift (PSI) and Net Benefit by month from data capture.")
    parser.add_argument("--processing-dir", type=Path, default=PROCESSING_DIR)
    parser.add_argument("--with-labels", action="store_true", help="join the late labels: Net Benefit by month")
    parser.add_argument("--publish-metrics", action="store_true", help="MaxPSI, NetBenefit, NetBenefitDropPct")
    args = parser.parse_args(argv)
    summary = summarize(args.processing_dir, args.with_labels)
    out = args.processing_dir / "output" / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "drift_report.html").write_text(html(summary))
    for month, v in summary["months"].items():
        print(f"{month}: rows {v.get('rows')}, max PSI {v.get('max_psi')} ({v.get('worst_feature')}, {v.get('band')}), "
              f"Net Benefit {v.get('net_benefit', '-')}, drop {v.get('net_benefit_drop_pct', '-')} %")
    if args.publish_metrics:
        for line in publish(summary):
            print("published", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
