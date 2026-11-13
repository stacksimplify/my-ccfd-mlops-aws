"""
MLOps 09 Step-02: the Net Benefit gate recommends; a person approves.

    python m09_02_review_package.py --version <n>

Candidate = package <n>; champion = the newest Approved package of ccfd-fraud-detector (none yet: the first one is
promoted). Both models are scored on the fixed gate holdout (2,000 rows, seed 301, threshold 0.5) with
mlops_utils/gate.py. It prints the decision and the status it suggests; the person who decides sets the status (with
their own reason) with aws sagemaker update-model-package. It changes nothing.
"""

import argparse

import m09_common as m09
from mlops_utils import endpoints, gate

parser = argparse.ArgumentParser(description="Gate one package against the newest Approved package.")
parser.add_argument("--version", required=True, type=int)
version = parser.parse_args().version

data = m09.gate_holdout()
candidate_pkg = m09.package(version)
candidate, _ = m09.tar_files(candidate_pkg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
newest = endpoints.newest_approved(m09.GROUP)
champion = champion_version = None
if newest is not None:
    champion_version = int(newest["ModelPackageVersion"])
    champion, _ = m09.tar_files(m09.package(champion_version)["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])

record = gate.run_gate(candidate, champion, data, candidate_name=m09.GROUP, candidate_version=str(version),
                       champion_name=m09.GROUP, champion_version=str(champion_version), data_name="gate holdout (seed 301)")
m09.say(f"Gate data: {record['gate_rows']:,} rows, {record['gate_frauds']} frauds, threshold {gate.THRESHOLD}")
if champion is None:
    m09.say(f"champion: none / candidate v{version}: ${record['candidate']['net_benefit']:,}")
    m09.say("Gate: PROMOTE (first champion)")
else:
    m09.say(f"champion v{champion_version}: ${record['champion']['net_benefit']:,} / candidate v{version}: "
            f"${record['candidate']['net_benefit']:,}")
    m09.say(f"Gate: {record['decision']} ({record['net_benefit_gain']:+,})")
m09.say(f"Suggested status: {'Approved' if record['promote'] else 'Rejected'} (you decide)")
