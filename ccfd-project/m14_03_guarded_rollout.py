"""MLOps 14 (cd.yml, after the GitHub Environment approval): set the package Approved, then a canary on
ccfd-prod with auto-rollback on the 2 rollout alarms. Exit 0 when ccfd-prod serves the new version, 1 when the
canary was rolled back (the package is then set Rejected, so it never ships by accident).
    python m14_03_guarded_rollout.py --package-version <N> (or --package-arn <ARN>) [--approved-by <github actor>]"""
import argparse
import sys

from mlops_utils import endpoints, release

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
which = parser.add_mutually_exclusive_group(required=True)
which.add_argument("--package-version", help="the version number (the workflows use this)")
which.add_argument("--package-arn")
parser.add_argument("--approved-by", default="unknown")
args = parser.parse_args()
args.package_arn = release.package_from_args(args)
version = release.version_of(args.package_arn)
release.set_status(args.package_arn, "Approved", f"cd.yml: GitHub Environment production approved by {args.approved_by}")
print(f"package version {version}: Approved")
serving = release.deploy_to_prod(version, log=endpoints._say)
if serving == version:
    print(f"ccfd-prod serves version {version}: rollout done")
    sys.exit(0)
release.set_status(args.package_arn, "Rejected", f"cd.yml: canary rolled back (ccfd-prod stayed on {serving})")
print(f"ROLLED BACK: ccfd-prod stays on version {serving}; package {version} set Rejected; @champion unchanged")
sys.exit(1)
