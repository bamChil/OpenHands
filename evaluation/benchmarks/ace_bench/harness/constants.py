"""Constants for ACE-Bench evaluation harness."""

from pathlib import Path

# Directory constants
ACE_BENCH_DIR = Path(__file__).parent.parent

# Docker constants
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"

# Log file names
LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
LOG_REPORT = "report.json"
LOG_PATCH = "patch.diff"

# Keys for prediction/instance data
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"

# Test result constants
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"

# Encoding
UTF8 = "utf-8"
