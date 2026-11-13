"""MLOps 14 (cd.yml, last step, only after a successful rollout): MLflow @champion follows the registry.
    python m14_03_sync_champion_alias.py --package-version <N> (or --package-arn <ARN>)"""
import argparse

from mlops_utils import endpoints, release

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
which = parser.add_mutually_exclusive_group(required=True)
which.add_argument("--package-version", help="the version number (the workflows use this)")
which.add_argument("--package-arn")
args = parser.parse_args()
args.package_arn = release.package_from_args(args)
release.sync_champion(args.package_arn, log=endpoints._say)
