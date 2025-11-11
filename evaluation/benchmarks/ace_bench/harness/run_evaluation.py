"""
ACE-Bench Evaluation Runner

This script runs evaluation for ACE-Bench predictions by:
1. Loading predictions from JSONL file
2. Loading dataset from HuggingFace
3. Creating Docker containers for each instance
4. Applying patches and running tests
5. Collecting and reporting results
"""

import argparse
import docker
import json
import logging
import os
import shutil
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from ..harness.constants import (
    ACE_BENCH_DIR,
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_PATCH,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    UTF8,
    DEFAULT_PYTEST_CMD,
)
from ..harness.test_parsers import MAP_REPO_TO_PARSER, MAP_REPO_TO_TEST_CMD
from ..harness.test_parsers import parse_log_pytest
from ..harness.utils import (
    EvaluationError,
    filter_predictions_by_ids,
    get_docker_image_name,
    get_instance_from_dataset,
    get_predictions_from_file,
    str2bool,
)
from ..harness.review_codes import (
    save_review_codes_level1,
    save_review_codes_level2,
)


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Set up logger for instance evaluation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def exec_run_with_timeout(
    container: docker.models.containers.Container,
    cmd: str,
    timeout: int | None = None,
    **kwargs
) -> tuple[int, bytes]:
    """
    Execute command in container with timeout.

    Args:
        container: Docker container
        cmd: Command to execute
        timeout: Timeout in seconds
        **kwargs: Additional arguments for container.exec_run

    Returns:
        Tuple of (exit_code, output)
    """
    try:
        result = container.exec_run(
            cmd,
            user=kwargs.get("user", DOCKER_USER),
            workdir=kwargs.get("workdir", DOCKER_WORKDIR),
            stream=False,
            demux=False,
            **{k: v for k, v in kwargs.items() if k not in ["user", "workdir"]}
        )
        return result.exit_code, result.output
    except Exception as e:
        return -1, str(e).encode(UTF8)


def copy_to_container(
    container: docker.models.containers.Container,
    src_path: str | Path,
    dst_path: str
) -> None:
    """
    Copy file to container.

    Args:
        container: Docker container
        src_path: Source file path on host
        dst_path: Destination path in container
    """
    import tarfile
    import io

    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Create tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(str(src_path), arcname=os.path.basename(dst_path))

    tar_stream.seek(0)

    # Put archive in container
    dst_dir = os.path.dirname(dst_path)
    container.put_archive(dst_dir, tar_stream)


def run_instance_level1(
    instance: pd.Series,
    pred: dict,
    container: docker.models.containers.Container,
    logger: logging.Logger,
    log_dir: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for Level 1 instance.

    Level 1 workflow:
    1. Activate conda environment
    2. Restore project from /root/my_repo/
    3. Apply patch (for masking) and directly delete F2P test files
    4. Reinitialize git
    5. Apply agent's patch
    6. Delete any generated F2P files, restore F2P test, and run tests

    Args:
        instance: Instance data (from HuggingFace)
        pred: Prediction data with patch
        container: Docker container
        logger: Logger instance
        log_dir: Directory to save test outputs
        timeout: Test timeout in seconds

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]
    logger.info(f"Starting Level 1 evaluation for {instance_id}")

    results = {
        "instance_id": instance_id,
        "level": 1,
        "patch_applied": False,
        "f2p_success": False,
        "p2p_success": False,
        "error": None,
    }

    try:
        # Step 1: Activate conda and restore project
        logger.info("Step 1: Activating conda environment and restoring project")
        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{cmd}'", timeout=600)
        logger.info(f"Restore project exit code: {exit_code}")
        logger.info(f"Output: {output.decode(UTF8, errors='replace')}")

        if exit_code != 0:
            raise EvaluationError(f"Failed to restore project: {output.decode(UTF8, errors='replace')}")

        # Step 2: Apply patch (for masking) and test_patch (for deleting F2P test)
        logger.info("Step 2: Applying patch to mask files")
        patch_content = instance.get('patch', '')

        if patch_content and patch_content.strip():
            # Write patch to temporary file on host
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(patch_content)
                temp_patch_path = f.name

            try:
                # Copy patch to container
                patch_path_container = "/tmp/mask_patch.diff"
                copy_to_container(container, temp_patch_path, patch_path_container)
                logger.info(f"Copied mask patch to container: {patch_path_container}")

                # Apply patch
                apply_cmd = f"cd /testbed && git apply --whitespace=fix {patch_path_container}"
                exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{apply_cmd}'", timeout=120)

                patch_output = output.decode(UTF8, errors='replace')
                logger.info(f"Mask patch apply exit code: {exit_code}")
                logger.info(f"Mask patch output: {patch_output}")

                if exit_code != 0:
                    logger.warning(f"Failed to apply mask patch: {patch_output}")
                else:
                    logger.info("Successfully applied mask patch")
            finally:
                os.unlink(temp_patch_path)
        else:
            logger.info("No patch to apply for masking")

        # Step 2b: Prepare test_patch for later use and delete F2P test files
        logger.info("Step 2b: Preparing test_patch and deleting F2P test files")

        # Save test_patch to container for later reverse apply (to restore test files)
        test_patch_content = instance.get('test_patch', '')
        test_patch_path_container = None

        if test_patch_content and test_patch_content.strip():
            # Write test_patch to temporary file on host
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(test_patch_content)
                temp_test_patch_path = f.name

            try:
                # Copy test_patch to container (for later reverse apply)
                test_patch_path_container = "/tmp/test_patch.diff"
                copy_to_container(container, temp_test_patch_path, test_patch_path_container)
                logger.info(f"Saved test_patch to container: {test_patch_path_container} (for later reverse apply)")
            finally:
                os.unlink(temp_test_patch_path)

        # Get F2P test paths from instance and delete them directly
        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if fail_to_pass:
            f2p_tests = fail_to_pass if isinstance(fail_to_pass, list) else [fail_to_pass]

            for f2p_test in f2p_tests:
                # Ensure path starts with /testbed/
                if not f2p_test.startswith('/testbed/'):
                    f2p_test_path = f'/testbed/{f2p_test}'
                else:
                    f2p_test_path = f2p_test

                logger.info(f"Deleting F2P test file: {f2p_test_path}")

                # Delete test file
                delete_cmd = f"rm -f {f2p_test_path}"
                exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{delete_cmd}'", timeout=60)

                if exit_code != 0:
                    logger.warning(f"Failed to delete F2P test file {f2p_test_path}: {output.decode(UTF8, errors='replace')}")
                else:
                    logger.info(f"Successfully deleted F2P test file: {f2p_test_path}")
        else:
            logger.warning("No FAIL_TO_PASS tests found in instance, skipping test file deletion")

        # Step 3: Reinitialize git repository
        logger.info("Step 3: Reinitializing git repository")
        git_cmds = [
            "cd /testbed && rm -rf .git",
            "cd /testbed && git init",
            'cd /testbed && git config user.email "ace@bench.com"',
            'cd /testbed && git config user.name "ACE Bench"',
            'cd /testbed && git add -A',
            'cd /testbed && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty',
        ]

        for cmd in git_cmds:
            exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{cmd}'", timeout=60)
            if exit_code != 0:
                logger.warning(f"Git command failed: {cmd}")
                logger.warning(f"Output: {output.decode(UTF8, errors='replace')}")

        # Step 4: Apply patch
        logger.info("Step 4: Applying agent patch")
        patch_content = pred[KEY_PREDICTION]

        if not patch_content or patch_content.strip() == "":
            logger.warning("Empty patch provided")
            results["error"] = "Empty patch"
            return results

        # Write patch to temporary file on host, then copy to container
        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False, encoding=UTF8) as f:
            f.write(patch_content)
            temp_patch_path = f.name

        try:
            # Copy patch file to container
            patch_path_container = "/tmp/agent_patch.diff"
            copy_to_container(container, temp_patch_path, patch_path_container)
            logger.info(f"Copied patch file to container: {patch_path_container}")
        finally:
            # Clean up temporary file
            os.unlink(temp_patch_path)

        # Apply patch with whitespace tolerance
        apply_cmd = f"cd /testbed && git apply --whitespace=fix --verbose {patch_path_container}"
        exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{apply_cmd}'", timeout=120)

        patch_output = output.decode(UTF8, errors='replace')
        logger.info(f"Patch apply exit code: {exit_code}")
        logger.info(f"Patch output: {patch_output}")

        if exit_code != 0:
            logger.error(f"{APPLY_PATCH_FAIL}")
            results["error"] = f"Patch apply failed: {patch_output}"
            return results

        logger.info(f"{APPLY_PATCH_PASS}")
        results["patch_applied"] = True

        # Step 5: Prepare for testing - delete generated F2P files and restore F2P test
        logger.info("Step 5: Preparing for testing - cleaning up and restoring F2P test")

        # Get F2P test paths from instance
        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if not fail_to_pass:
            logger.error("No FAIL_TO_PASS tests found in instance")
            results["error"] = "No FAIL_TO_PASS tests"
            return results

        f2p_test_path = fail_to_pass[0] if isinstance(fail_to_pass, list) else fail_to_pass
        # Ensure path starts with /testbed/
        if not f2p_test_path.startswith('/testbed/'):
            f2p_test_path = f"/testbed/{f2p_test_path}"

        logger.info(f"F2P test path: {f2p_test_path}")

        # Delete F2P test file if it exists (may have been generated by agent's patch)
        logger.info(f"Checking if {f2p_test_path} exists and deleting if present")
        delete_cmd = f"rm -f {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{delete_cmd}'", timeout=60)
        logger.info(f"Delete F2P file exit code: {exit_code}")

        # Restore F2P test file using reverse apply test_patch
        if test_patch_path_container:
            logger.info("Reverse applying test_patch to restore F2P test file")
            reverse_cmd = f"cd /testbed && git apply --reverse --whitespace=fix {test_patch_path_container}"
            exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{reverse_cmd}'", timeout=120)

            reverse_output = output.decode(UTF8, errors='replace')
            logger.info(f"Reverse apply exit code: {exit_code}")
            logger.info(f"Reverse apply output: {reverse_output}")

            if exit_code != 0:
                logger.warning(f"Failed to reverse apply test_patch: {reverse_output}")
            else:
                logger.info("Successfully restored F2P test file using test_patch")
        else:
            logger.warning("No test_patch available to restore F2P test file")

        # Step 6: Run tests
        logger.info("Step 6: Running tests")
        repo_name = instance.get("repo_name", "")
        test_cmd = MAP_REPO_TO_TEST_CMD.get(repo_name, DEFAULT_PYTEST_CMD)

        # Get P2P tests
        pass_to_pass = instance.get('PASS_TO_PASS', [])
        p2p_tests = pass_to_pass if isinstance(pass_to_pass, list) else []

        # Run F2P test
        logger.info(f"Running F2P test: {f2p_test_path}")
        test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(
            container, f"/bin/bash -c '{test_cmd_full}'", timeout=timeout or 1800
        )

        f2p_output = output.decode(UTF8, errors='replace')
        logger.info(f"F2P test exit code: {exit_code}")
        logger.info(f"F2P output (truncated): {f2p_output[:500]}")

        results["f2p_success"] = (exit_code == 0)

        # Save F2P test output to file (像 SWE-Bench 一样)
        test_output_file = log_dir / LOG_TEST_OUTPUT
        with open(test_output_file, "w", encoding=UTF8) as f:
            f.write(f2p_output)
        logger.info(f"Saved F2P test output to {test_output_file}")

        # Run P2P tests
        if p2p_tests:
            logger.info(f"Running {len(p2p_tests)} P2P tests")
            p2p_results = []
            for p2p_test in p2p_tests:
                # Ensure path starts with /testbed/
                p2p_test_path = p2p_test
                if not p2p_test_path.startswith('/testbed/'):
                    p2p_test_path = f"/testbed/{p2p_test_path}"

                logger.info(f"Running P2P test: {p2p_test_path}")

                test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {p2p_test_path}"
                exit_code, output = exec_run_with_timeout(
                    container, f"/bin/bash -c '{test_cmd_full}'", timeout=timeout or 1800
                )

                p2p_output = output.decode(UTF8, errors='replace')
                logger.info(f"P2P test {p2p_test_path} exit code: {exit_code}")

                p2p_results.append(exit_code == 0)

                # Save P2P test output to file
                # Extract test file name from path (e.g., /testbed/test/transformers/test_cross_entropy.py -> test_cross_entropy.py)
                test_file_name = os.path.basename(p2p_test_path).replace('.py', '')
                p2p_output_file = log_dir / f"test_output_p2p_{test_file_name}.txt"
                with open(p2p_output_file, "w", encoding=UTF8) as f:
                    f.write(p2p_output)
                logger.info(f"Saved P2P test output to {p2p_output_file}")

            results["p2p_success"] = all(p2p_results)
        else:
            results["p2p_success"] = True  # No P2P tests means success

        return results

    except Exception as e:
        logger.error(f"Error in Level 1 evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results


def run_instance_level2(
    instance: pd.Series,
    pred: dict,
    container: docker.models.containers.Container,
    logger: logging.Logger,
    log_dir: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for Level 2 instance.

    Level 2 workflow:
    1. Activate conda environment
    2. Clean /testbed/ and initialize git
    3. Apply agent's patch
    4. Install agent's implementation (pip install .)
    5. Restore original project from /root/my_repo/
    6. Copy masked test files
    7. Run F2P tests only

    Args:
        instance: Instance data
        pred: Prediction data with patch
        container: Docker container
        logger: Logger instance
        log_dir: Directory to save test outputs
        timeout: Test timeout in seconds

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]
    logger.info(f"Starting Level 2 evaluation for {instance_id}")

    results = {
        "instance_id": instance_id,
        "level": 2,
        "patch_applied": False,
        "install_success": False,
        "f2p_success": False,
        "error": None,
    }

    try:
        # Step 1: Activate conda environment
        logger.info("Step 1: Activating conda environment")
        cmd = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{cmd}'", timeout=60)

        if exit_code != 0:
            raise EvaluationError(f"Failed to activate conda: {output.decode(UTF8, errors='replace')}")

        # Step 2: Clean /testbed/ and initialize git
        logger.info("Step 2: Cleaning /testbed/ and initializing git")
        init_cmds = [
            "rm -rf /testbed/* /testbed/.*  2>/dev/null || true",
            "mkdir -p /testbed",
            "cd /testbed && git init",
            'cd /testbed && git config user.email "ace@bench.com"',
            'cd /testbed && git config user.name "ACE Bench"',
            'cd /testbed && echo "put all codes in this folder" > README.md',
            'cd /testbed && git add -A',
            'cd /testbed && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty',
        ]

        for cmd in init_cmds:
            exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{cmd}'", timeout=60)
            if exit_code != 0 and "rm -rf" not in cmd:  # rm -rf is allowed to fail
                logger.warning(f"Init command failed: {cmd}")
                logger.warning(f"Output: {output.decode(UTF8, errors='replace')}")

        # Step 3: Apply patch
        logger.info("Step 3: Applying agent patch")
        patch_content = pred[KEY_PREDICTION]

        if not patch_content or patch_content.strip() == "":
            logger.warning("Empty patch provided")
            results["error"] = "Empty patch"
            return results

        # Write patch to temporary file on host, then copy to container
        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False, encoding=UTF8) as f:
            f.write(patch_content)
            temp_patch_path = f.name

        try:
            # Copy patch file to container
            patch_path_container = "/tmp/agent_patch.diff"
            copy_to_container(container, temp_patch_path, patch_path_container)
            logger.info(f"Copied patch file to container: {patch_path_container}")
        finally:
            # Clean up temporary file
            os.unlink(temp_patch_path)

        # Apply patch with whitespace tolerance
        apply_cmd = f"cd /testbed && git apply --whitespace=fix --verbose {patch_path_container}"
        exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{apply_cmd}'", timeout=120)

        patch_output = output.decode(UTF8, errors='replace')
        logger.info(f"Patch apply exit code: {exit_code}")
        logger.info(f"Patch output: {patch_output}")

        if exit_code != 0:
            logger.error(f"{APPLY_PATCH_FAIL}")
            results["error"] = f"Patch apply failed: {patch_output}"
            return results

        logger.info(f"{APPLY_PATCH_PASS}")
        results["patch_applied"] = True

        # Step 4: Install agent's implementation
        logger.info("Step 4: Installing agent's implementation")
        install_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "cd /testbed && "
            "pip install ."
        )
        exit_code, output = exec_run_with_timeout(
            container, f"/bin/bash -c '{install_cmd}'", timeout=600
        )

        install_output = output.decode(UTF8, errors='replace')
        logger.info(f"Install exit code: {exit_code}")
        logger.info(f"Install output (truncated): {install_output[:500]}")

        if exit_code != 0:
            logger.warning(f"Installation failed, but continuing: {install_output}")
            # Don't fail here, continue to test
        else:
            results["install_success"] = True

        # Step 5: Restore original project
        logger.info("Step 5: Restoring original project")
        restore_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(
            container, f"/bin/bash -c '{restore_cmd}'", timeout=600
        )

        if exit_code != 0:
            raise EvaluationError(f"Failed to restore project: {output.decode(UTF8, errors='replace')}")

        # Step 6: Apply test_patch to modify test files
        logger.info("Step 6: Applying test_patch to modify test files")
        test_patch_content = instance.get('test_patch', '')

        if test_patch_content and test_patch_content.strip():
            # Write test_patch to temporary file on host
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(test_patch_content)
                temp_test_patch_path = f.name

            try:
                # Copy test_patch to container
                test_patch_path_container = "/tmp/test_patch.diff"
                copy_to_container(container, temp_test_patch_path, test_patch_path_container)
                logger.info(f"Copied test_patch to container: {test_patch_path_container}")

                # Apply test_patch
                apply_cmd = f"cd /testbed && git apply --whitespace=fix {test_patch_path_container}"
                exit_code, output = exec_run_with_timeout(container, f"/bin/bash -c '{apply_cmd}'", timeout=120)

                test_patch_output = output.decode(UTF8, errors='replace')
                logger.info(f"Test patch apply exit code: {exit_code}")
                logger.info(f"Test patch output: {test_patch_output}")

                if exit_code != 0:
                    logger.warning(f"Failed to apply test_patch: {test_patch_output}")
                else:
                    logger.info("Successfully applied test_patch")
            finally:
                os.unlink(temp_test_patch_path)
        else:
            logger.info("No test_patch to apply")

        # Step 7: Run F2P test
        logger.info("Step 7: Running F2P tests")
        repo_name = instance.get("repo_name", "")
        test_cmd = MAP_REPO_TO_TEST_CMD.get(repo_name, DEFAULT_PYTEST_CMD)

        # Get F2P test path from instance
        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if not fail_to_pass:
            logger.error("No FAIL_TO_PASS tests found in instance")
            results["error"] = "No FAIL_TO_PASS tests"
            return results

        f2p_test = fail_to_pass[0] if isinstance(fail_to_pass, list) else fail_to_pass
        # Ensure path starts with /testbed/
        f2p_test_path = f2p_test
        if not f2p_test_path.startswith('/testbed/'):
            f2p_test_path = f"/testbed/{f2p_test_path}"

        # Run F2P test
        logger.info(f"Running F2P test: {f2p_test_path}")
        test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(
            container, f"/bin/bash -c '{test_cmd_full}'", timeout=timeout or 1800
        )

        f2p_output = output.decode(UTF8, errors='replace')
        logger.info(f"F2P test exit code: {exit_code}")
        logger.info(f"F2P output (truncated): {f2p_output[:500]}")

        results["f2p_success"] = (exit_code == 0)

        # Save F2P test output to file (像 SWE-Bench 一样)
        test_output_file = log_dir / LOG_TEST_OUTPUT
        with open(test_output_file, "w", encoding=UTF8) as f:
            f.write(f2p_output)
        logger.info(f"Saved F2P test output to {test_output_file}")

        return results

    except Exception as e:
        logger.error(f"Error in Level 2 evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results


def run_instance(
    instance: pd.Series,
    pred: dict,
    output_dir: Path,
    timeout: int | None = None,
    gpu_ids: str | None = None,
    review_codes: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation for a single instance.

    Args:
        instance: Instance data from ACE-Bench dataset
        pred: Prediction dictionary with patch
        output_dir: Output directory for evaluation results (from predictions_path)
        timeout: Test timeout in seconds
        gpu_ids: Comma-separated GPU IDs to use (e.g., '0,1' or '2,3')
        review_codes: Whether to save agent-generated code for review

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]

    # Setup logging directory - save to eval_outputs/instance_id/ in output_dir
    log_dir = output_dir / "eval_outputs" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    logger.info(f"{'=' * 60}")
    logger.info(f"Starting evaluation for instance: {instance_id}")
    logger.info(f"Level: {instance['level']}")
    logger.info(f"{'=' * 60}")

    # Check if report already exists
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        logger.info("Report already exists, skipping evaluation")
        with open(report_path, "r", encoding=UTF8) as f:
            existing_report = json.load(f)
        # Return in the format expected by summary generation
        return {
            "instance_id": instance_id,
            "completed": True,
            "resolved": existing_report.get(instance_id, {}).get("resolved", False),
            "patch_applied": existing_report.get(instance_id, {}).get("patch_successfully_applied", False),
            "report": existing_report,
        }

    container = None
    try:
        # Get Docker image name
        docker_image = get_docker_image_name(instance)
        logger.info(f"Using Docker image: {docker_image}")

        # Create Docker client and container
        client = docker.from_env()

        # Pull image if not exists
        try:
            client.images.get(docker_image)
            logger.info(f"Image {docker_image} found locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {docker_image}")
            client.images.pull(docker_image)

        # Create container with GPU support
        import time
        timestamp = int(time.time())
        container_name = f"ace_eval_{instance_id}_{timestamp}".replace("/", "_").replace("__", "_")
        logger.info(f"Creating container: {container_name}")

        # Configure GPU access
        if gpu_ids is not None:
            # Use specific GPU IDs
            device_requests = [
                docker.types.DeviceRequest(device_ids=gpu_ids.split(','), capabilities=[['gpu']])
            ]
            logger.info(f"GPU access requested for specific GPUs: {gpu_ids}")
            nvidia_visible_devices = gpu_ids
        else:
            # Use all available GPUs
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
            logger.info("GPU access requested for all available GPUs")
            nvidia_visible_devices = 'all'

        # Set environment variables for NVIDIA GPU support
        environment = {
            'NVIDIA_VISIBLE_DEVICES': nvidia_visible_devices,
            'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility',
        }

        try:
            container = client.containers.run(
                docker_image,
                command="/bin/bash -c 'sleep infinity'",
                name=container_name,
                detach=True,
                remove=False,
                user=DOCKER_USER,
                working_dir=DOCKER_WORKDIR,
                device_requests=device_requests,
                environment=environment,
            )
        except Exception as e:
            logger.error(f"Failed to create container with GPU support: {e}")
            logger.info("If NVIDIA Docker runtime is not available, please install it first")
            raise

        logger.info(f"Container {container_name} created successfully")

        # Run evaluation based on level
        level = int(instance["level"])
        if level == 1:
            results = run_instance_level1(instance, pred, container, logger, log_dir, timeout)
        elif level == 2:
            results = run_instance_level2(instance, pred, container, logger, log_dir, timeout)
        else:
            raise ValueError(f"Unsupported level: {level}")

        # Save patch
        patch_path = log_dir / LOG_PATCH
        with open(patch_path, "w", encoding=UTF8) as f:
            f.write(pred.get(KEY_PREDICTION, ""))

        # Parse test outputs and generate structured report
        patch_content = pred.get(KEY_PREDICTION, "")
        patch_is_none = patch_content is None
        patch_exists = bool(patch_content and patch_content.strip())
        patch_applied = results.get("patch_applied", False)

        # Get repo name for parser selection
        repo_name = instance.get("repo_name", "")
        parser_fn = MAP_REPO_TO_PARSER.get(repo_name, parse_log_pytest)

        # Parse F2P test output
        f2p_output_file = log_dir / LOG_TEST_OUTPUT
        f2p_parsed = {"tests": {}, "pass_rate": 0.0}
        if f2p_output_file.exists():
            with open(f2p_output_file, "r", encoding=UTF8, errors='replace') as f:
                f2p_output = f.read()
            f2p_parsed = parser_fn(f2p_output)

        # Parse P2P test outputs (Level 1 only)
        p2p_parsed_list = []
        if level == 1:
            # Find all P2P test output files
            for p2p_file in log_dir.glob("test_output_p2p_*.txt"):
                with open(p2p_file, "r", encoding=UTF8, errors='replace') as f:
                    p2p_output = f.read()
                p2p_parsed_list.append(parser_fn(p2p_output))

        # Build tests_status structure
        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        pass_to_pass = instance.get('PASS_TO_PASS', [])

        # For F2P tests
        f2p_success = []
        f2p_failure = []
        for test_name, test_status in f2p_parsed.get('tests', {}).items():
            if test_status in ['PASSED', 'SKIPPED', 'XFAIL', 'WARNING']:
                f2p_success.append(test_name)
            else:
                f2p_failure.append(test_name)

        # For P2P tests
        p2p_success = []
        p2p_failure = []
        for p2p_parsed in p2p_parsed_list:
            for test_name, test_status in p2p_parsed.get('tests', {}).items():
                if test_status in ['PASSED', 'SKIPPED', 'XFAIL', 'WARNING']:
                    p2p_success.append(test_name)
                else:
                    p2p_failure.append(test_name)

        # Determine if resolved
        resolved = results.get("f2p_success", False) and results.get("p2p_success", True)

        # Calculate F2P pass rate (only FAIL_TO_PASS tests, not including PASS_TO_PASS)
        f2p_total = len(f2p_success) + len(f2p_failure)
        f2p_pass_rate = round(len(f2p_success) / f2p_total, 4) if f2p_total > 0 else 0.0

        # Generate report in the required format
        report = {
            instance_id: {
                "patch_is_None": patch_is_none,
                "patch_exists": patch_exists,
                "patch_successfully_applied": patch_applied,
                "resolved": resolved,
                "pass_rate": f2p_pass_rate,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": f2p_success,
                        "failure": f2p_failure
                    },
                    "PASS_TO_PASS": {
                        "success": p2p_success,
                        "failure": p2p_failure
                    }
                }
            }
        }

        # Save report
        with open(report_path, "w", encoding=UTF8) as f:
            json.dump(report, f, indent=4)

        logger.info(f"{'=' * 60}")
        logger.info(f"Evaluation completed for {instance_id}")
        logger.info(f"Resolved: {resolved}")
        logger.info(f"F2P Pass rate: {f2p_pass_rate}")
        logger.info(f"{'=' * 60}")

        # Save review codes if requested
        if review_codes:
            logger.info("Saving agent-generated code for review...")
            docker_image = get_docker_image_name(instance)
            if level == 1:
                save_review_codes_level1(instance, patch_content, log_dir, docker_image, logger)
            elif level == 2:
                save_review_codes_level2(instance, patch_content, log_dir, docker_image, logger)
            logger.info("Review codes saved.")

        return {
            "instance_id": instance_id,
            "completed": True,
            "resolved": resolved,
            "patch_applied": patch_applied,
            "report": report,
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())

        # Save error report in the required format
        patch_content = pred.get(KEY_PREDICTION, "")
        error_report = {
            instance_id: {
                "patch_is_None": patch_content is None,
                "patch_exists": bool(patch_content and patch_content.strip()),
                "patch_successfully_applied": False,
                "resolved": False,
                "pass_rate": 0.0,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": [],
                        "failure": []
                    },
                    "PASS_TO_PASS": {
                        "success": [],
                        "failure": []
                    }
                }
            }
        }

        with open(log_dir / LOG_REPORT, "w", encoding=UTF8) as f:
            json.dump(error_report, f, indent=4)

        return {
            "instance_id": instance_id,
            "completed": True,
            "resolved": False,
            "patch_applied": False,
            "error": str(e),
            "report": error_report,
        }

    finally:
        # Cleanup container
        if container is not None:
            try:
                logger.info(f"Stopping and removing container")
                container.stop(timeout=10)
                container.remove()
                logger.info(f"Container removed successfully")
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")


def main():
    """Main entry point for ACE-Bench evaluation."""
    parser = argparse.ArgumentParser(
        description="Run ACE-Bench evaluation on predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        help="Specific instance IDs to evaluate (optional)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout for test execution (seconds)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1' or '2,3'). If not specified, all GPUs will be available.",
    )
    parser.add_argument(
        "--review-codes",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
        help="Save agent-generated code for review after evaluation (true/false)",
    )

    args = parser.parse_args()

    # Get output directory from predictions_path
    predictions_path = Path(args.predictions_path)
    output_dir = predictions_path.parent
    print(f"Output directory: {output_dir}")

    # Load dataset from HuggingFace
    print("Loading ACE-Bench dataset from HuggingFace...")

    # Get HuggingFace token (if private dataset)
    hf_token = os.environ.get('HF_TOKEN', None)
    if hf_token:
        print('Using HuggingFace token from HF_TOKEN environment variable')

    # Clear cache to avoid corruption
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/BamChil___ace-bench')
    if os.path.exists(cache_dir):
        print(f'Found existing cache at {cache_dir}, removing...')
        try:
            shutil.rmtree(cache_dir)
            print('Cache cleared successfully')
        except Exception as e:
            print(f'Warning: Failed to clear cache: {e}')

    try:
        # Load both Level 1 and Level 2 datasets
        dataset_lv1 = load_dataset("BamChil/ACE-Bench", split="level1", token=hf_token)
        df_lv1 = pd.DataFrame(dataset_lv1)
        df_lv1['level'] = 1
        print(f"Loaded {len(df_lv1)} Level 1 instances")

        dataset_lv2 = load_dataset("BamChil/ACE-Bench", split="level2", token=hf_token)
        df_lv2 = pd.DataFrame(dataset_lv2)
        df_lv2['level'] = 2
        print(f"Loaded {len(df_lv2)} Level 2 instances")

        # Combine datasets
        dataset = pd.concat([df_lv1, df_lv2], ignore_index=True)
        print(f"Total: {len(dataset)} instances")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Troubleshooting:")
        print("1. Set HF_TOKEN: export HF_TOKEN=your_token")
        print("2. Check access: https://huggingface.co/datasets/BamChil/ACE-Bench")
        raise

    print(f"Loading predictions from {args.predictions_path}...")
    predictions = get_predictions_from_file(args.predictions_path)
    print(f"Loaded {len(predictions)} predictions")

    # Filter by instance IDs if specified
    if args.instance_ids:
        predictions = filter_predictions_by_ids(predictions, args.instance_ids)
        print(f"Filtered to {len(predictions)} predictions")

    # Match predictions with dataset
    instances_to_eval = []
    for pred in predictions:
        instance_id = pred[KEY_INSTANCE_ID]
        instance = get_instance_from_dataset(dataset, instance_id)
        if instance is None:
            print(f"Warning: Instance {instance_id} not found in dataset")
            continue
        instances_to_eval.append((instance, pred))

    print(f"\nEvaluating {len(instances_to_eval)} instances...")
    print(f"Max workers: {args.max_workers}")
    print(f"Timeout: {args.timeout}s")
    if args.gpu_ids:
        print(f"Using GPUs: {args.gpu_ids}")
    else:
        print(f"Using all available GPUs")
    if args.review_codes:
        print("Review codes: ENABLED (will save agent code for review)")
    print()

    # Run evaluations
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_instance,
                instance,
                pred,
                output_dir,
                args.timeout,
                args.gpu_ids,
                args.review_codes,
            ): instance[KEY_INSTANCE_ID]
            for instance, pred in instances_to_eval
        }

        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                resolved_str = "✓" if result.get("resolved") else "✗"
                print(f"{resolved_str} {instance_id}: {result.get('completed', False)}")
            except Exception as e:
                print(f"✗ {instance_id}: ERROR - {str(e)}")
                results.append({
                    "instance_id": instance_id,
                    "completed": False,
                    "resolved": False,
                    "patch_applied": False,
                    "error": str(e),
                })

    # Generate summary report
    print(f"\n{'=' * 60}")
    print("Generating summary report...")
    print(f"{'=' * 60}")

    # Calculate statistics
    total_instances = len(results)
    completed_instances = sum(1 for r in results if r.get('completed', False))
    resolved_instances = sum(1 for r in results if r.get('resolved', False))
    unresolved_instances = completed_instances - resolved_instances
    empty_patch_instances = sum(1 for r in results if not r.get('patch_applied', False))
    error_instances = sum(1 for r in results if r.get('error'))

    # Collect IDs
    completed_ids = [r['instance_id'] for r in results if r.get('completed', False)]
    submitted_ids = [r['instance_id'] for r in results]
    resolved_ids = [r['instance_id'] for r in results if r.get('resolved', False)]
    unresolved_ids = [r['instance_id'] for r in results if r.get('completed', False) and not r.get('resolved', False)]
    incomplete_ids = [r['instance_id'] for r in results if not r.get('completed', False)]
    empty_patch_ids = [r['instance_id'] for r in results if not r.get('patch_applied', False)]
    error_ids = [r['instance_id'] for r in results if r.get('error')]

    # Calculate rates
    resolved_rate = round(resolved_instances / total_instances, 4) if total_instances > 0 else 0.0

    # Calculate average F2P pass rate (average of all individual F2P pass rates)
    pass_rates = []
    for result in results:
        if 'report' in result:
            for instance_id, instance_report in result['report'].items():
                pass_rates.append(instance_report.get('pass_rate', 0.0))
    average_f2p_pass_rate = round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0.0

    # Generate summary report
    summary_report = {
        "total_instances": total_instances,
        "submitted_instances": len(submitted_ids),
        "completed_instances": completed_instances,
        "resolved_instances": resolved_instances,
        "unresolved_instances": unresolved_instances,
        "empty_patch_instances": empty_patch_instances,
        "error_instances": error_instances,
        "resolved_rate": resolved_rate,
        "pass_rate": average_f2p_pass_rate,
        "submitted_ids": submitted_ids,
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids
    }

    # Save summary report to output directory
    summary_report_path = output_dir / "report.json"
    with open(summary_report_path, "w", encoding=UTF8) as f:
        json.dump(summary_report, f, indent=4)

    print(f"Summary report saved to: {summary_report_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Total instances: {total_instances}")
    print(f"Completed: {completed_instances}")
    print(f"Resolved: {resolved_instances}")
    print(f"Unresolved: {unresolved_instances}")
    print(f"Empty patch: {empty_patch_instances}")
    print(f"Errors: {error_instances}")
    print(f"Resolved rate: {resolved_rate * 100:.1f}%")
    print(f"Average F2P pass rate: {average_f2p_pass_rate * 100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

