"""
MLOps 12 Step-01: the 2 rollout alarms on ccfd-prod (their definitions live in mlops_utils/endpoints.py).

    python m12_01_rollout_alarms.py create     ccfd-prod-rollout-5xx (any 5xx) and -netbenefit (< 33,000)
    python m12_01_rollout_alarms.py delete
"""

import sys

import m12_common as m12
from mlops_utils import config, endpoints

if (sys.argv[1] if len(sys.argv) > 1 else "create") == "create":
    endpoints.create_rollout_alarms(endpoint=m12.EP, tags=config.load().aws_tags(), log=m12.log)
else:
    endpoints.delete_rollout_alarms(log=m12.log)
