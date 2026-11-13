"""
MLOps 15: split each simulated month into what the endpoint sees and what arrives later.

    python m15_01_split_months.py [--tag replay1]

Reads data/production_month_1.csv .. _4.csv (made by m11_02_generate_months.py) and writes, per month:
  data/months/month_<k>_requests.jsonl   one transaction per line (the 8 request fields, NO label) + its inference_id
  data/months/month_<k>_labels.csv       inference_id,is_fraud: the true labels, uploaded later (late labels)
inference_id = <tag>:month_<k>:<row_id, 6 digits>, so captured predictions join with the labels.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mlops_utils.monitoring import FIELDS

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--tag", default="replay1")
args = parser.parse_args()
out = Path("data/months")
out.mkdir(parents=True, exist_ok=True)
for k in (1, 2, 3, 4):
    month = pd.read_csv(f"data/production_month_{k}.csv")
    ids = [f"{args.tag}:month_{k}:{i:06d}" for i in range(1, len(month) + 1)]
    with open(out / f"month_{k}_requests.jsonl", "w") as f:
        f.writelines(json.dumps({"inference_id": inference_id, "transaction": record}) + "\n" for inference_id, record in zip(ids, month[FIELDS].to_dict("records"), strict=True))
    pd.DataFrame({"inference_id": ids, "is_fraud": month["is_fraud"]}).to_csv(out / f"month_{k}_labels.csv",
                                                                              index=False)
    print(f"month_{k}: {len(month):,} requests, {int(month['is_fraud'].sum())} frauds -> data/months/month_{k}_*")
print(f"first inference_id: {args.tag}:month_1:000001")
