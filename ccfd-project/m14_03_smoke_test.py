"""MLOps 14 (cd.yml): the base project's 3 test transactions against an endpoint; exit 1 unless 3 answers come back.
    python m14_03_smoke_test.py [--endpoint ccfd-staging]"""
import argparse
import json
import sys
from pathlib import Path

from mlops_utils import endpoints, release

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--endpoint", default=release.STAGING)
args = parser.parse_args()
body = json.loads(Path("tests/payloads/sample_transactions.json").read_text())
answer = endpoints.invoke(args.endpoint, body, log=endpoints._say)       # one retry inside
results = json.loads(answer["body"])
ok = isinstance(results, list) and len(results) == len(body) and all("prediction" in r for r in results)
for r in results if isinstance(results, list) else []:
    print(f"  {r.get('prediction')} {r.get('probability', 0):.3f} ({r.get('model_name')})")
print(f"smoke test on {args.endpoint}: {'PASS' if ok else 'FAIL'} ({answer['attempts']} attempt(s))")
sys.exit(0 if ok else 1)
