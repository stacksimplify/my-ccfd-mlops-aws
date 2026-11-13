"""
Test settings shared by every test in this folder.

1. Where the parity test (tests/test_parity_p5_04.py) sends its 2,000 rows:
     --url http://localhost:8080   a running server or container (or set PARITY_URL)
     --endpoint ccfd-rt            a SageMaker endpoint, called with boto3 and your AWS profile
                                   (or set PARITY_ENDPOINT); costs money, so only in a paid session
     neither                       the serving API inside this Python process (no server needed)
   Example: python -m pytest -s tests/test_parity_p5_04.py --url http://localhost:8080

2. MLOPS_REQUIRE_MODEL=1 (CI and the build checks): the laptop data and model must be there. Without them the
   run stops at once with an error, instead of skipping the tests that need them.
   Example: MLOPS_REQUIRE_MODEL=1 python -m pytest -q

3. The Docker image checks (Docker must run; the images are built with --load, never pushed):
     MLOPS_SERVING_IMAGE=ccfd-serving:dev     the tar tests also start the real serving image (/ping 200)
     MLOPS_TRAINING_IMAGE=ccfd-training:dev   the training image runs its 3 entrypoints and the train launcher
   Without a variable those checks are skipped, and the summary says so ("skipped"; -rs gives the reason).
   MLOPS_REQUIRE_IMAGE=1 turns a missing variable into a failure. A step that changes an image, the model files or
   the packager runs:
     MLOPS_SERVING_IMAGE=ccfd-serving:dev MLOPS_TRAINING_IMAGE=ccfd-training:dev MLOPS_REQUIRE_IMAGE=1 \
         MLOPS_REQUIRE_MODEL=1 python -m pytest -q -rs
"""

import os
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
FE_DATA = PROJECT / "data" / "credit_card_transactions_fe1_amount_ratio.csv"
MODEL_FILE = PROJECT / "models" / "p5_03_production" / "model_latest.pkl"


def pytest_addoption(parser):
    parser.addoption("--url", default=os.environ.get("PARITY_URL"),
                     help="parity test: the serving API at this URL, for example http://localhost:8080")
    parser.addoption("--endpoint", default=os.environ.get("PARITY_ENDPOINT"),
                     help="parity test: this SageMaker endpoint, for example ccfd-rt (paid sessions only)")


@pytest.fixture
def docker_image():
    """
    docker_image("MLOPS_SERVING_IMAGE", "ccfd-serving:dev") gives the image named in that variable. Without it the
    test is skipped with the reason, or fails when MLOPS_REQUIRE_IMAGE=1, so a run never looks as if it checked an
    image it did not start.
    """
    def image_from(variable: str, example: str) -> str:
        image = os.environ.get(variable, "").strip()
        if not image:
            message = f"image not checked: set {variable}={example} (Docker must run)"
            if os.environ.get("MLOPS_REQUIRE_IMAGE") == "1":
                pytest.fail(f"MLOPS_REQUIRE_IMAGE=1, but {message}")
            pytest.skip(message)
        return image
    return image_from


def pytest_configure(config):
    if config.getoption("--url") and config.getoption("--endpoint"):
        raise pytest.UsageError("Give --url or --endpoint, not both.")
    if os.environ.get("MLOPS_REQUIRE_MODEL") == "1":
        missing = [str(path.relative_to(PROJECT)) for path in (FE_DATA, MODEL_FILE) if not path.exists()]
        if missing:
            raise pytest.UsageError(f"MLOPS_REQUIRE_MODEL=1, but these files are missing: {', '.join(missing)}. "
                                    "Run the base project's chain p1_01, p4_01 ... p5_03 first.")
