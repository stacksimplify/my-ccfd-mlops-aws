"""MLOps 14: do the two registries agree? The SageMaker registry decides what ships (ccfd-prod's package) and
MLflow @champion should point at the same package. They are 2 separate writes, so a release can be half done.
    python m14_04_check_sync.py      exit 0 IN SYNC, 1 HALF DONE (with the line to fix it)"""
import sys

from mlops_utils import release

prod, champion = release.prod_version(), release.champion_package_version()
# One fact per line, each with its label
print(f"ccfd-prod serves:  package version {prod}")
print(f"MLflow @champion:  package_version {champion}")
if prod == champion:
    print("IN SYNC")
    sys.exit(0)
print("HALF DONE: run python m14_03_sync_champion_alias.py --package-arn <the ARN of version "
      f"{prod}> (or rollback.yml)")
sys.exit(1)
