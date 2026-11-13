"""
The course settings in one place.

The MLflow tracking URI comes ONLY from the environment variable MLFLOW_TRACKING_URI (MLOps 04 rule:
"scripts read only MLFLOW_TRACKING_URI"):
  - not set: the local MLflow server of MLOps 04 (Step-01-02), http://127.0.0.1:7080;
  - on AWS you set it yourself from the App ARN that create-env.sh wrote into .env.mlops (MLOps 07):
        export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN"
    A .env.mlops file alone never sends your runs to AWS, and the ARN is never built from the App name.

The MLOPS_* names (bucket name, suffix, App ARN, tag values), first match wins:
  1. the environment (an empty value counts as not set)
  2. the .env.mlops file (written by scripts/create-env.sh, never by hand, never committed)
  3. the default in this file, if there is one

Usage:
    from mlops_utils import config
    settings = config.load()
    print(settings.tracking_uri)       # http://127.0.0.1:7080 on the laptop
    print(settings.bucket("jobs"))     # ccfd-mlops-jobs-<MLOPS_SUFFIX>, or None before create-env.sh
    print(settings.tags)               # the course tags for every AWS resource (needs .env.mlops)

Only the standard library is used, so this file also runs inside the training and serving images.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed names from the names table (a section never invents its own)
# ---------------------------------------------------------------------------
REGION = "us-east-1"                                 # every course resource
DEFAULT_TRACKING_URI = "http://127.0.0.1:7080"       # macOS keeps ports 5000 and 7000
TRACKING_URI_VARIABLE = "MLFLOW_TRACKING_URI"        # the one variable that picks the MLflow server (MLOps 04, 07)
ENV_FILE_NAME = ".env.mlops"
ENV_PREFIX = "MLOPS_"                                   # every .env.mlops variable starts with MLOPS_
# The student's lab repo, the default of --dvc-repo in the student scripts that need DVC (MLOps 03-05).
# Checks and tests pass --dvc-repo themselves. expanduser never fails, even in an image without a home folder.
LAB_REPO = Path(os.path.expanduser("~/mlops-lab/ccfd-project"))

EXPERIMENT_NAME = "ccfd-fraud-detection"
REGISTERED_MODEL = "ccfd-fraud-model"                # MLflow registry only
ALIAS_CHAMPION = "champion"                          # the model in production
ALIAS_CANDIDATE = "candidate"                        # the model the gate checks next
MODEL_PACKAGE_GROUP = "ccfd-fraud-detector"          # SageMaker registry: written by boto3 only
MLFLOW_APP_NAME = "ccfd-mlflow"
EXECUTION_ROLE_NAME = "ccfd-sagemaker-execution-role"
ECR_SERVING_REPO = "ccfd-serving"
ECR_TRAINING_REPO = "ccfd-training"
SNS_TOPIC = "ccfd-mlops-alerts"
BUCKET_PURPOSES = ("tfstate", "dvc", "mlflow", "jobs")  # ccfd-mlops-<purpose>-<random6>

COURSE_TAG = "mlops-on-aws"                          # the value of the course tag, never changed
# owner, purpose and created are set once in .env.mlops by create-env.sh; no default on purpose.
TAG_VARIABLES = {"owner": "MLOPS_OWNER", "purpose": "MLOPS_PURPOSE", "created": "MLOPS_CREATED"}

# The App ARN has the form arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-<id>
APP_ARN_PATTERN = re.compile(r"^arn:aws:sagemaker:([a-z0-9-]+):\d{12}:mlflow-app/app-[A-Za-z0-9-]+$")
SUFFIX_PATTERN = re.compile(r"^[0-9a-f]{6}$")        # random_id byte_length 3, as .hex
ACCOUNT_ID_PATTERN = re.compile(r"\d{12}")
KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def mask(text: str) -> str:
    """Hide every 12-digit number (an AWS account ID) before text is printed or logged."""
    return ACCOUNT_ID_PATTERN.sub("<ACCOUNT_ID>", str(text))


def _env_value(raw: str) -> str:
    """One value the way bash 'source' reads it: the quotes removed, a ' # comment' after the value left out."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end > 0:
            return raw[1:end]                        # the text inside the quotes; what follows them is a comment
    return re.split(r"\s#", raw, maxsplit=1)[0].rstrip()


def read_env_file(path: Path) -> dict[str, str]:
    """
    Read a simple KEY=VALUE file such as .env.mlops.

    Accepted lines:   KEY=value   export KEY=value   KEY="value"   KEY='value'   KEY=value  # a comment
    Ignored lines:    empty lines, lines that start with #, lines without '=', keys that are not a variable name.
    """
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = re.sub(r"^export\s+", "", line)
        key, value = line.split("=", 1)
        key = key.strip()
        if KEY_PATTERN.match(key):
            values[key] = _env_value(value)
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Look for .env.mlops in the start folder (default: the current folder) and then in each parent folder."""
    folder = Path(start or Path.cwd()).resolve()
    for candidate in (folder, *folder.parents):
        env_file = candidate / ENV_FILE_NAME
        if env_file.is_file():
            return env_file
    return None


@dataclass(frozen=True, repr=False)
class Settings:
    """All course settings. Make one with load(); it never changes after that."""

    values: dict = field(default_factory=dict)      # MLFLOW_TRACKING_URI + the MLOPS_* values (never printed)
    env_file: Path | None = None                     # the .env.mlops file that was read, if any

    def __repr__(self) -> str:
        """Only the variable NAMES, never a value: a Settings object in a log or a test failure leaks nothing."""
        return f"Settings(variables={sorted(self.values)}, env_file={mask(self.env_file)})"

    def get(self, key: str, default: str | None = None) -> str | None:
        """One setting: the environment, then .env.mlops, then the default."""
        value = self.values.get(key)
        return value if value else default

    @property
    def region(self) -> str:
        return REGION

    @property
    def tracking_uri(self) -> str:
        """MLFLOW_TRACKING_URI from the environment, else the local server http://127.0.0.1:7080."""
        uri = self.get(TRACKING_URI_VARIABLE)
        if uri is None:
            return DEFAULT_TRACKING_URI
        if uri.startswith("arn:"):
            check_app_arn(uri, TRACKING_URI_VARIABLE)
        return uri

    @property
    def app_arn(self) -> str | None:
        """
        MLOPS_MLFLOW_APP_ARN, the ARN of the SageMaker MLflow App that create-env.sh copied from 'terraform output -raw'
        (checked), or None. Nothing uses it on its own: a script that goes to the App exports it as MLFLOW_TRACKING_URI.
        """
        value = self.get("MLOPS_MLFLOW_APP_ARN")
        if value is not None:
            check_app_arn(value, "MLOPS_MLFLOW_APP_ARN")
        return value

    @property
    def suffix(self) -> str | None:
        """MLOPS_SUFFIX: the 6 hex characters create-env.sh makes once (Terraform random_id)."""
        value = self.get("MLOPS_SUFFIX")
        if value is not None and not SUFFIX_PATTERN.match(value):
            raise ValueError("MLOPS_SUFFIX must be 6 hex characters (Terraform random_id byte_length 3, as .hex)")
        return value

    def bucket(self, purpose: str) -> str | None:
        """
        A bucket name: ccfd-mlops-<purpose>-<MLOPS_SUFFIX>, or None before create-env.sh has run.
        For the jobs bucket, MLOPS_JOBS_BUCKET wins when it is set (the name MLOps 10 and 12 use).
        """
        if purpose not in BUCKET_PURPOSES:
            raise ValueError(f"Unknown bucket purpose '{purpose}'. Use one of: {', '.join(BUCKET_PURPOSES)}")
        if purpose == "jobs" and self.get("MLOPS_JOBS_BUCKET") is not None:
            return self.get("MLOPS_JOBS_BUCKET")
        return f"ccfd-mlops-{purpose}-{self.suffix}" if self.suffix else None

    @property
    def tags(self) -> dict[str, str]:
        """The course tags for every AWS resource: course is fixed; owner, purpose and created come from .env.mlops."""
        missing = [name for name in TAG_VARIABLES.values() if self.get(name) is None]
        if missing:
            raise ValueError(f"{', '.join(missing)} not set. The course tags come from .env.mlops: run "
                             "scripts/create-env.sh first (it writes them once), or export them.")
        return {"course": COURSE_TAG, **{tag: self.get(name) for tag, name in TAG_VARIABLES.items()}}

    def aws_tags(self) -> list[dict[str, str]]:
        """The same tags in the form boto3 create calls want: [{"Key": ..., "Value": ...}, ...]."""
        return [{"Key": key, "Value": value} for key, value in self.tags.items()]


def check_app_arn(value: str, name: str) -> None:
    """Stop with a clear message when a value is not an MLflow App ARN in us-east-1. The value is never printed."""
    match = APP_ARN_PATTERN.match(value)
    if not match:
        raise ValueError(
            f"{name} is not an MLflow App ARN. Expected arn:aws:sagemaker:{REGION}:<account>:mlflow-app/app-<id>, "
            "copied by create-env.sh from 'terraform output -raw' (never built from the App name)."
        )
    if match.group(1) != REGION:
        raise ValueError(f"{name} points to an MLflow App outside {REGION}; this course uses {REGION} only.")


def load(env_file: Path | str | bool | None = None, environ: dict | None = None) -> Settings:
    """
    Read the settings once.

    env_file: None (the default) = the first .env.mlops in this folder or a parent folder;
              a path = that file (skipped if it does not exist); False = no .env.mlops at all (the tests use it)
    environ:  the environment to use (default: os.environ); tests pass their own dict

    Only MLFLOW_TRACKING_URI (from the environment only) and the MLOPS_* variables are kept; nothing else of the
    environment (for example AWS keys) is ever stored. An empty value counts as not set, so an empty exported
    variable never hides the .env.mlops line.
    """
    environ = os.environ if environ is None else environ
    if env_file is False:
        path = None
    elif env_file is None:
        path = find_env_file()
    else:
        path = Path(env_file)
    from_file = read_env_file(path) if path is not None and path.is_file() else {}

    values = {key: value for key, value in from_file.items() if key.startswith(ENV_PREFIX) and value != ""}
    # The environment goes on top, so an exported variable beats the same line in .env.mlops
    values.update({key: value for key, value in environ.items()
                   if (key == TRACKING_URI_VARIABLE or key.startswith(ENV_PREFIX)) and value != ""})
    return Settings(values=values, env_file=path if path is not None and path.is_file() else None)
