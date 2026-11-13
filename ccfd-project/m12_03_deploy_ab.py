"""
MLOps 12 Step-03: the A/B mechanics: champion (package 1) x 1 with weight 0.8, challenger x 1 with weight 0.2.
The challenger is the drill model of Step-02 (the only other model in the registry); in real work it is a new
candidate that passed the gate.

    python m12_03_deploy_ab.py
"""

import m12_common as m12
from mlops_utils import endpoints

m12.make_config("ccfd-prod-config-ab", [m12.variant("champion", "ccfd-prod-model-v1", 1, 0.8),
                                        m12.variant("challenger", "ccfd-prod-model-drill", 1, 0.2)])
if not m12.already_serves("ccfd-prod-config-ab"):
    endpoints.update(m12.EP, "ccfd-prod-config-ab", log=m12.log, timeout_seconds=3600)
m12.say(f"{m12.EP}: InService on {m12.current_config()}")
