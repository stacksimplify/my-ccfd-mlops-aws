"""
MLOps 01 Step-03: evaluate any endpoint on the base project's 2,000 test transactions.

Does the deployed model earn what training promised? The same 2,000 transactions, any endpoint:
    python m01_03_evaluate_endpoint.py --url http://localhost:8001    the base project's API (POST /inference per row)
    python m01_03_evaluate_endpoint.py --url http://localhost:8080    the serving API or a container (POST /invocations)
    python m01_03_evaluate_endpoint.py --endpoint ccfd-rt             a SageMaker endpoint (from MLOps 10)
It prints caught / missed / false alarms / Net Benefit and compares the Net Benefit with the number the model saved at
training time (models/<model_name>/metadata_latest.json), or with --expect. Exit 0 = PASS, 1 = FAIL or error.
Run it from the ccfd-project folder (it needs data/ from p1_01 and p4_01).
"""

import argparse
import sys

from mlops_utils import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an endpoint on the base project's 2,000 test transactions.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="a laptop API or a container, for example http://localhost:8080")
    where.add_argument("--endpoint", help="a SageMaker endpoint name, for example ccfd-rt")
    parser.add_argument("--expect", type=int, help="the expected Net Benefit (default: the model's own metadata)")
    args = parser.parse_args()

    try:
        records, labels = evaluate.test_set()
        if args.url:
            answers, how, model_name = evaluate.answers_from_url(args.url, records)
        else:
            answers, how, model_name = evaluate.answers_from_endpoint(args.endpoint, records)
        result = evaluate.score(labels, answers)
    except evaluate.EvaluationError as err:
        print(f"ERROR: {err}")
        return 1

    if args.expect is not None:
        expected, source = args.expect, "--expect"
    else:
        expected, source = evaluate.expected_net_benefit(model_name)

    print(f"Endpoint: {args.url or args.endpoint} ({how}), model {model_name}")
    print(f"Test set: {len(labels):,} transactions, {sum(labels)} frauds (the base project's split, seed 42)")
    print(f"Caught {result['caught']}, missed {result['missed']}, false alarms {result['false_alarms']:,}")
    if expected is None:
        print(f"Net Benefit: ${result['net_benefit']:,}   expected: unknown ({source})   NO CHECK")
        return 1
    verdict = "PASS" if result["net_benefit"] == expected else "FAIL"
    print(f"Net Benefit: ${result['net_benefit']:,}   expected ({source}): ${expected:,}   {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
