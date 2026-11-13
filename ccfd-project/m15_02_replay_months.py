"""
MLOps 15: replay the simulated months against ccfd-prod, one transaction per call, each with its InferenceId.

    python m15_02_replay_months.py [--months 1 2 3] [--threads 8] [--dry-run]

Reads data/months/month_<k>_requests.jsonl (m15_01). Every call goes through endpoints.invoke() (one retry) with
InferenceId <tag>:month_<k>:<row_id>, so the capture record joins with the late label. The first capture file lands
in S3 about 5 minutes later.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlops_utils import endpoints

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--months", type=int, nargs="+", default=[1, 2, 3, 4])
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--dry-run", action="store_true", help="print the first InferenceId of each month, call nothing")
args = parser.parse_args()
for k in args.months:
    lines = [json.loads(x) for x in Path(f"data/months/month_{k}_requests.jsonl").read_text().splitlines()]
    if args.dry_run:
        print(f"month_{k}: {len(lines):,} requests, first {lines[0]['inference_id']}")
        continue
    start, failed = time.time(), 0

    def call(line):
        endpoints.invoke(endpoints.PROD_ENDPOINT, [line["transaction"]], inference_id=line["inference_id"],
                         log=lambda _text: None)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for future in [pool.submit(call, line) for line in lines]:
            try:
                future.result()
            except Exception:  # noqa: BLE001 (counted and reported; the replay goes on)
                failed += 1
    print(f"month_{k}: {len(lines):,} calls in {time.time() - start:.0f} s, {failed} failed")
