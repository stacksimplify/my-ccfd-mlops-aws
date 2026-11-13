"""
MLOps 11 (BONUS): the Lambda behind the public URL. It passes the JSON body to the serverless endpoint and its answer
back; the serving container checks and scores (m01_03_serving_api.py), so the answer is the one of POST /invocations.

    client --HTTPS--> API Gateway HTTP API (POST /inference) --> this Lambda --> invoke_endpoint --> ccfd-serverless

Lambda settings: handler m11_04_lambda_handler.lambda_handler; ENDPOINT_NAME (default ccfd-serverless).
"""

import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "ccfd-serverless")
RUNTIME = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def lambda_handler(event, context):
    for _ in range(2):   # a cold serverless endpoint can fail its first call: try once more
        try:
            answer = RUNTIME.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="application/json",
                                             Accept="application/json", Body=(event.get("body") or "").encode())
            return {"statusCode": 200, "headers": {"Content-Type": "application/json"},
                    "body": answer["Body"].read().decode()}
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "Error")
        except BotoCoreError as error:
            code = type(error).__name__
    return {"statusCode": 502, "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"detail": f"The model endpoint could not be called ({code})."})}
