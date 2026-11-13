"""
MLOps 11 Step-04 (BONUS): a public HTTPS URL in front of ccfd-serverless: API Gateway HTTP API -> Lambda.

    python m11_04_api.py up       role ccfd-inference-proxy-role, function ccfd-inference-proxy, API ccfd-inference-api
    python m11_04_api.py test     the base project's 3 transactions through the URL
    python m11_04_api.py down     delete all 3 (destroy-env.sh also deletes them)

The URL is public while it exists and has no login: delete it in the same session. The Lambda may only call endpoints
named ccfd-*. Production would add a login: IAM auth, or Cognito through a JWT authorizer (not built here).
AWS WAF works only with a REST API, not with this HTTP API.
The same names, settings and policy text as the commands of the README (Option 1). The role policy says * for the
account: a SageMaker endpoint and a log group can only be reached from their own account, so the role still reaches
only this account's ccfd- endpoints and its own log group.
"""

import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config

R, NAME, API, ROLE = config.REGION, "ccfd-inference-proxy", "ccfd-inference-api", "ccfd-inference-proxy-role"
iam, lam, api = boto3.client("iam"), boto3.client("lambda", region_name=R), boto3.client("apigatewayv2", region_name=R)


def say(text) -> None:
    print(config.mask(str(text)), flush=True)


def find_api():
    """The API ccfd-inference-api (its ApiId and ApiEndpoint), or None."""
    return next((item for item in api.get_apis()["Items"] if item["Name"] == API), None)


def up() -> None:
    try:   # Option 1 (or an earlier up) made it: change nothing
        iam.get_role(RoleName=ROLE)
        say(f"STOP: {ROLE} exists already (Option 1 or an earlier up made it); nothing was changed. "
            "Call the URL: python m11_04_api.py test. Delete it: python m11_04_api.py down")
        sys.exit(1)
    except iam.exceptions.NoSuchEntityException:
        pass
    tags = config.load().aws_tags()
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                                                      "Action": "sts:AssumeRole"}]}
    role = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Tags=tags)["Role"]["Arn"]
    account = role.split(":")[4]   # arn:aws:iam::<account>:role/...
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{ROLE}-policy", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": "sagemaker:InvokeEndpoint", "Resource": f"arn:aws:sagemaker:{R}:*:endpoint/ccfd-*"},
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": f"arn:aws:logs:{R}:*:log-group:/aws/lambda/{NAME}*"}]}))
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write("m11_04_lambda_handler.py")
    for attempt in range(10):   # a new role needs a few seconds before Lambda can use it
        try:
            fn = lam.create_function(FunctionName=NAME, Runtime="python3.13", Role=role, Timeout=29, MemorySize=256,
                                     Handler="m11_04_lambda_handler.lambda_handler", Code={"ZipFile": zipped.getvalue()},
                                     Environment={"Variables": {"ENDPOINT_NAME": "ccfd-serverless"}},
                                     Tags={t["Key"]: t["Value"] for t in tags})["FunctionArn"]
            break
        except lam.exceptions.InvalidParameterValueException:
            if attempt == 9:
                raise
            time.sleep(6)
    lam.get_waiter("function_active_v2").wait(FunctionName=NAME)
    http = api.create_api(Name=API, ProtocolType="HTTP", Target=fn, RouteKey="POST /inference",
                          Tags={t["Key"]: t["Value"] for t in tags})
    lam.add_permission(FunctionName=NAME, StatementId=API, Action="lambda:InvokeFunction",
                       Principal="apigateway.amazonaws.com",
                       SourceArn=f"arn:aws:execute-api:{R}:{account}:{http['ApiId']}/*/*/inference")
    say(f"API up: {http['ApiEndpoint']}/inference (public, no login: delete it in this session)")


def post(url: str, transaction: dict) -> tuple:
    """POST one transaction; a new API answers 404 for a few seconds until its $default stage is deployed."""
    request = urllib.request.Request(url, data=json.dumps(transaction).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(15):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read()), time.perf_counter() - start
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == 14:
                raise
            if attempt == 0:
                say("HTTP 404: the new API is not deployed yet; trying again every 2 s")
            time.sleep(2)


def test() -> None:
    found = find_api()
    if not found:
        say(f"STOP: no API {API}: make it first with: python m11_04_api.py up")
        sys.exit(1)
    url = found["ApiEndpoint"] + "/inference"
    for transaction in json.loads(Path("tests/payloads/sample_transactions.json").read_text()):
        status, body, seconds = post(url, transaction)
        body = body[0] if isinstance(body, list) else body
        say(f"HTTP {status} in {seconds:.2f} s: {body['prediction']} "
            f"{100 * body['probability']:.2f}% {body['risk_score']}")


def down() -> None:
    found = find_api()
    if found:
        api.delete_api(ApiId=found["ApiId"])
    for call in (lambda: lam.delete_function(FunctionName=NAME),
                 lambda: iam.delete_role_policy(RoleName=ROLE, PolicyName=f"{ROLE}-policy"),
                 lambda: iam.delete_role(RoleName=ROLE),
                 lambda: boto3.client("logs", region_name=R).delete_log_group(logGroupName=f"/aws/lambda/{NAME}")):
        try:
            call()
        except ClientError:
            pass   # already gone
    say("API down: the API, the function, its role and its log group are deleted")


{"up": up, "test": test, "down": down}[sys.argv[1] if len(sys.argv) > 1 else "test"]()
