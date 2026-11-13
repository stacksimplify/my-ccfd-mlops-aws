"""
MLOps 12 Step-04 (BONUS): a shadow variant: the champion answers every call; the shadow (the drill model of
Step-02) gets a copy of each request, and its answers are never returned. Only the routing is shown here; a team
also captures the shadow's answers (data capture, MLOps 15) and compares them with the champion's for days.

    python m12_04_deploy_shadow.py
"""

import m12_common as m12
from mlops_utils import endpoints

m12.make_config("ccfd-prod-config-shadow", [m12.variant("champion", "ccfd-prod-model-v1", 1)],
                shadow=[m12.variant("shadow-drill", "ccfd-prod-model-drill", 1)])
if not m12.already_serves("ccfd-prod-config-shadow"):
    endpoints.update(m12.EP, "ccfd-prod-config-shadow", log=m12.log, timeout_seconds=3600)
answered = {m12.score()[:2] for _ in range(10)}
described = m12.sm.describe_endpoint(EndpointName=m12.EP)
m12.say(f"10 calls answered by: {sorted(answered)}")
m12.say("shadow variants: " + ", ".join(f"{v['VariantName']} x{v.get('CurrentInstanceCount')}"
                                       for v in described.get("ShadowProductionVariants", [])))
