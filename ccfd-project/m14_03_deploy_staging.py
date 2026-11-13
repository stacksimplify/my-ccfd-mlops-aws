"""MLOps 14 (cd.yml): put a still-Pending package on ccfd-staging as a PLAIN model (its Image and ModelDataUrl,
never the package itself: a Pending package cannot be deployed as a package).
    python m14_03_deploy_staging.py --package-version <N>   (or --package-arn <ARN>)"""
import argparse

from mlops_utils import endpoints, release

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
which = parser.add_mutually_exclusive_group(required=True)
which.add_argument("--package-version", help="the version number (the workflows use this)")
which.add_argument("--package-arn")
args = parser.parse_args()
args.package_arn = release.package_from_args(args)
version = release.version_of(args.package_arn)
endpoints.deploy_package(release.STAGING, version, instance_count=1, plain=True, log=endpoints._say)
print(f"{release.STAGING}: package version {version} InService (plain model)")
