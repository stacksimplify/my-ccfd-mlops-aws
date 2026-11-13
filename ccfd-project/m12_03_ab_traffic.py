"""
MLOps 12 Step-03: send N requests (each = the 2,000-row test set); count them and their Net Benefit per variant.

    python m12_03_ab_traffic.py --requests 100
    python m12_03_ab_traffic.py --requests 1 --target-variant challenger
"""

import argparse

import m12_common as m12

parser = argparse.ArgumentParser(description="A/B traffic on ccfd-prod.")
parser.add_argument("--requests", type=int, default=100)
parser.add_argument("--target-variant")
args = parser.parse_args()
counts = {}
for _ in range(args.requests):
    variant, model, net_benefit = m12.score(args.target_variant)
    counts.setdefault((variant, model), []).append(net_benefit)
for (variant, model), values in sorted(counts.items()):
    m12.say(f"{variant}: {len(values)} of {args.requests} requests, {model}, Net Benefit ${values[0]:,}")
