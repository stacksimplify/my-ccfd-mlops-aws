"""MLOps 14 (rollback.yml): the previous Approved package back on ccfd-prod (a direct update, no canary: a
rollback must be quick), the current one set Rejected, and @champion moved back.
    python m14_04_rollback.py"""
import sys

from mlops_utils import endpoints, release

current = release.prod_version()
if current is None:
    sys.exit("STOP: there is no ccfd-prod to roll back")
earlier = [v for v in endpoints.approved_versions() if v < current]
if not earlier:
    sys.exit(f"STOP: no Approved version older than {current}")
previous = earlier[-1]
serving = release.deploy_to_prod(previous, canary=False, log=endpoints._say)
release.set_status(endpoints.package_arn(current), "Rejected", f"rollback.yml: replaced by version {previous}")
release.sync_champion(endpoints.package_arn(previous), log=endpoints._say)
# One fact per line, each with its label
print(f"rolled back ccfd-prod:   version {current} -> {serving}")
print(f"rolled back package {current}:  Rejected")
print(f"rolled back @champion:   package {previous}")
