"""
Tests for mlops_utils/config.py.

MLFLOW_TRACKING_URI comes from the environment only (default: the local server); for the MLOPS_* names the environment
beats .env.mlops and .env.mlops beats the defaults. Every test passes its own environment and its own .env.mlops (or
env_file=False), so a real .env.mlops on the laptop (create-env.sh writes one) never changes a result.

No 12-digit number is written in this file: the made-up account number is built when the test runs.
"""

import re

import pytest

from mlops_utils import config

FAKE_ACCOUNT = "".join(str((7 * i + 3) % 10) for i in range(12))      # made up at run time
APP_ARN = f"arn:aws:sagemaker:us-east-1:{FAKE_ACCOUNT}:mlflow-app/app-ABC123def"
TAGS = {"MLOPS_OWNER": "kalyan", "MLOPS_PURPOSE": "a0-build", "MLOPS_CREATED": "2026-10-12"}


def write_env_file(folder, text):
    path = folder / ".env.mlops"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_env_file_formats(tmp_path):
    path = write_env_file(tmp_path, "# written by create-env.sh\n\n"
                                    "MLOPS_SUFFIX=a1b2c3\n"
                                    "export MLOPS_JOBS_BUCKET=ccfd-mlops-jobs-a1b2c3\n"
                                    f"export\tMLOPS_MLFLOW_APP_ARN={APP_ARN}\n"
                                    'MLOPS_OWNER="kalyan"\n'
                                    "MLOPS_PURPOSE='a0-build'\n"
                                    "not a setting\n"
                                    "not a key=value\n")
    assert config.read_env_file(path) == {"MLOPS_SUFFIX": "a1b2c3", "MLOPS_JOBS_BUCKET": "ccfd-mlops-jobs-a1b2c3",
                                          "MLOPS_MLFLOW_APP_ARN": APP_ARN, "MLOPS_OWNER": "kalyan", "MLOPS_PURPOSE": "a0-build"}


def test_read_env_file_inline_comments_like_bash(tmp_path):
    """bash 'source' drops a ' # comment' after a value; a # inside a value or inside quotes stays."""
    path = write_env_file(tmp_path, "MLOPS_OWNER=kalyan # the owner tag\n"
                                    'MLOPS_PURPOSE="a0 build" # x\n'
                                    "MLOPS_CREATED='2026-10-12'   # y\n"
                                    "NOTE=a#b\n"
                                    'QUOTED="a # b"\n')
    assert config.read_env_file(path) == {"MLOPS_OWNER": "kalyan", "MLOPS_PURPOSE": "a0 build", "MLOPS_CREATED": "2026-10-12",
                                          "NOTE": "a#b", "QUOTED": "a # b"}


def test_defaults_without_env_file_or_variables():
    settings = config.load(env_file=False, environ={})
    assert settings.tracking_uri == "http://127.0.0.1:7080"
    assert settings.region == "us-east-1"
    assert settings.bucket("jobs") is None                 # create-env.sh has not run yet
    assert settings.app_arn is None and settings.env_file is None


def test_tracking_uri_comes_only_from_the_environment(tmp_path):
    """MLOps 04: scripts read only MLFLOW_TRACKING_URI. The App ARN in .env.mlops never moves runs to AWS alone."""
    path = write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={APP_ARN}\nMLFLOW_TRACKING_URI={APP_ARN}\nMLOPS_SUFFIX=a1b2c3\n")
    settings = config.load(env_file=path, environ={})
    assert settings.tracking_uri == "http://127.0.0.1:7080"   # a fresh shell: the local server, not the App
    assert settings.app_arn == APP_ARN                         # the checked ARN, for scripts that export it
    assert settings.bucket("dvc") == "ccfd-mlops-dvc-a1b2c3"
    assert settings.env_file == path
    # MLOps 07: export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN" is the one change
    assert config.load(env_file=path, environ={"MLFLOW_TRACKING_URI": APP_ARN}).tracking_uri == APP_ARN


def test_environment_beats_env_file(tmp_path):
    path = write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={APP_ARN}\nMLOPS_SUFFIX=a1b2c3\nMLOPS_OWNER=from-file\n")
    environ = {"MLFLOW_TRACKING_URI": "http://127.0.0.1:7080", "MLOPS_SUFFIX": "dddddd", "MLOPS_OWNER": "from-env",
               "MLOPS_PURPOSE": "a0-build", "MLOPS_CREATED": "2026-10-12"}
    settings = config.load(env_file=path, environ=environ)
    assert settings.tracking_uri == "http://127.0.0.1:7080"
    assert settings.suffix == "dddddd"
    assert settings.tags["owner"] == "from-env"


def test_empty_value_counts_as_not_set(tmp_path):
    path = write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={APP_ARN}\nMLOPS_SUFFIX=\n")
    settings = config.load(env_file=path, environ={"MLFLOW_TRACKING_URI": "", "MLOPS_MLFLOW_APP_ARN": ""})
    assert settings.tracking_uri == "http://127.0.0.1:7080"
    assert settings.app_arn == APP_ARN                     # an empty exported variable does not hide the file line
    assert settings.suffix is None


def test_app_name_is_refused_and_never_printed(tmp_path):
    for bad in ("ccfd-mlflow", f"arn:aws:sagemaker:us-west-2:{FAKE_ACCOUNT}:mlflow-app/app-ABC123def",
                f"arn:aws:sagemaker:us-east-1:{FAKE_ACCOUNT}:mlflow-tracking-server/ccfd-mlflow"):
        settings = config.load(env_file=write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={bad}\n"), environ={})
        with pytest.raises(ValueError) as error:
            _ = settings.app_arn
        assert FAKE_ACCOUNT not in str(error.value) and "ccfd-mlflow" not in str(error.value)
        if bad.startswith("arn:"):
            with pytest.raises(ValueError) as error:
                _ = config.load(env_file=False, environ={"MLFLOW_TRACKING_URI": bad}).tracking_uri
            assert FAKE_ACCOUNT not in str(error.value)


def test_buckets_and_suffix():
    settings = config.load(env_file=False, environ={"MLOPS_SUFFIX": "a1b2c3", "MLOPS_JOBS_BUCKET": "my-jobs-bucket"})
    assert settings.bucket("jobs") == "my-jobs-bucket"     # MLOPS_JOBS_BUCKET (12) wins over the suffix
    assert [settings.bucket(p) for p in ("tfstate", "dvc", "mlflow")] == \
        ["ccfd-mlops-tfstate-a1b2c3", "ccfd-mlops-dvc-a1b2c3", "ccfd-mlops-mlflow-a1b2c3"]   # from MLOPS_SUFFIX
    with pytest.raises(ValueError):
        settings.bucket("models")
    with pytest.raises(ValueError):
        _ = config.load(env_file=False, environ={"MLOPS_SUFFIX": "XYZ"}).suffix


def test_tags_n01():
    settings = config.load(env_file=False, environ=TAGS)
    assert settings.tags == {"course": "mlops-on-aws", "owner": "kalyan", "purpose": "a0-build",
                             "created": "2026-10-12"}
    assert settings.aws_tags()[0] == {"Key": "course", "Value": "mlops-on-aws"}


def test_tags_have_no_default():
    """No build-time value (purpose=a0-build, today's date) may reach a student copy: missing tags stop with a clear error."""
    for missing in TAGS:
        environ = {key: value for key, value in TAGS.items() if key != missing}
        with pytest.raises(ValueError, match=missing):
            _ = config.load(env_file=False, environ=environ).tags


def test_find_env_file_walks_up(tmp_path):
    path = write_env_file(tmp_path, "MLOPS_SUFFIX=a1b2c3\n")
    nested = tmp_path / "ccfd-project" / "mlops_utils"
    nested.mkdir(parents=True)
    assert config.find_env_file(nested) == path


def test_no_env_file_means_none_is_read(tmp_path, monkeypatch):
    """env_file=False reads no .env.mlops, even with one in a parent folder (as on the laptop after create-env.sh)."""
    write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={APP_ARN}\nMLOPS_SUFFIX=fedcba\n")
    nested = tmp_path / "600_app_source" / "ccfd-project"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert config.load(environ={}).suffix == "fedcba"                 # the default: search this folder and parents
    no_file = config.load(env_file=False, environ={})
    assert (no_file.suffix, no_file.app_arn, no_file.env_file) == (None, None, None)
    assert config.load(env_file=tmp_path / "none.env.mlops", environ={}).suffix is None   # a missing file is skipped


def test_repr_and_values_leak_nothing(tmp_path):
    """A Settings object in a log or a failed assert shows variable names only, never a value."""
    fake_secret = "FAKE-SECRET-" + "VALUE"
    path = write_env_file(tmp_path, f"MLOPS_MLFLOW_APP_ARN={APP_ARN}\n")
    settings = config.load(env_file=path, environ={"AWS_SECRET_ACCESS_KEY": fake_secret, "AWS_SESSION_TOKEN": "tok",
                                                   "MLOPS_SUFFIX": "a1b2c3", "HOME": "/x"})
    assert set(settings.values) == {"MLOPS_MLFLOW_APP_ARN", "MLOPS_SUFFIX"}   # nothing else of the environment is kept
    text = repr(settings) + str(settings)
    assert "MLOPS_MLFLOW_APP_ARN" in text and "MLOPS_SUFFIX" in text
    assert fake_secret not in text and "a1b2c3" not in text and "ABC123def" not in text
    assert not re.search(r"\d{12}", text)


def test_mask_hides_account_ids():
    assert config.mask(APP_ARN) == "arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:mlflow-app/app-ABC123def"


def test_lab_repo_is_n19():
    assert config.LAB_REPO.parts[-2:] == ("mlops-lab", "ccfd-project")
