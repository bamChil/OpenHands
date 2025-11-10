"""Utility functions for ACE-Bench evaluation harness."""

import json
import pandas as pd
from pathlib import Path
from typing import Any

from ..harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


def load_ace_bench_dataset() -> pd.DataFrame:
    """
    Load ACE-Bench dataset from HuggingFace (deprecated - kept for compatibility).

    This function is deprecated. Use load_dataset from datasets library instead.

    Returns:
        DataFrame containing ACE-Bench instances
    """
    # This function is kept for backward compatibility
    # New code should load directly from HuggingFace
    from datasets import load_dataset
    import os

    hf_token = os.environ.get('HF_TOKEN', None)

    # Load from HuggingFace
    dataset_lv1 = load_dataset("BamChil/ACE-Bench", split="level1", token=hf_token)
    df_lv1 = pd.DataFrame(dataset_lv1)
    df_lv1['level'] = 1

    dataset_lv2 = load_dataset("BamChil/ACE-Bench", split="level2", token=hf_token)
    df_lv2 = pd.DataFrame(dataset_lv2)
    df_lv2['level'] = 2

    df = pd.concat([df_lv1, df_lv2], ignore_index=True)
    return df


def get_predictions_from_file(file_path: str | Path) -> list[dict]:
    """
    Load predictions from a JSONL file.

    Each line should be a JSON object with at least:
    - instance_id: str
    - model_patch: str (or test_result.git_patch)
    - model_name_or_path: str (optional)

    Args:
        file_path: Path to predictions JSONL file

    Returns:
        List of prediction dictionaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")

    predictions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                # Normalize prediction format
                if KEY_PREDICTION not in pred:
                    # Try to extract from test_result.git_patch
                    if "test_result" in pred and "git_patch" in pred["test_result"]:
                        pred[KEY_PREDICTION] = pred["test_result"]["git_patch"]
                    else:
                        pred[KEY_PREDICTION] = ""

                # Ensure instance_id exists
                if KEY_INSTANCE_ID not in pred:
                    raise ValueError(f"Prediction missing '{KEY_INSTANCE_ID}': {pred}")

                # Set default model name if not provided
                if KEY_MODEL not in pred:
                    pred[KEY_MODEL] = "unknown"

                predictions.append(pred)

    return predictions


def filter_predictions_by_ids(
    predictions: list[dict],
    instance_ids: list[str] | None = None
) -> list[dict]:
    """
    Filter predictions by instance IDs.

    Args:
        predictions: List of predictions
        instance_ids: List of instance IDs to keep (None = keep all)

    Returns:
        Filtered list of predictions
    """
    if instance_ids is None:
        return predictions

    instance_ids_set = set(instance_ids)
    return [p for p in predictions if p[KEY_INSTANCE_ID] in instance_ids_set]


def get_instance_from_dataset(
    dataset: pd.DataFrame,
    instance_id: str
) -> pd.Series | None:
    """
    Get a single instance from the dataset by instance_id.

    Args:
        dataset: ACE-Bench dataset DataFrame
        instance_id: Instance ID to retrieve

    Returns:
        Instance as pandas Series, or None if not found
    """
    matches = dataset[dataset[KEY_INSTANCE_ID] == instance_id]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def get_docker_image_name(instance: pd.Series) -> str:
    """
    Get Docker image name for an instance.

    Args:
        instance: Instance data as pandas Series (from HuggingFace)

    Returns:
        Docker image name (e.g., docker.io/bamchil/image_name)
    """
    # HuggingFace dataset uses 'image_name' field
    if "image_name" in instance and pd.notna(instance["image_name"]):
        image_name = instance["image_name"]
        # If doesn't have registry prefix, add docker.io
        if '/' not in image_name or not image_name.startswith(('docker.io/', 'gcr.io/', 'ghcr.io/')):
            return f"docker.io/{image_name}".lower()
        return image_name.lower()

    # Fallback to old instance_image field for backward compatibility
    if "instance_image" not in instance or pd.isna(instance["instance_image"]):
        raise ValueError(f"image_name not found for {instance[KEY_INSTANCE_ID]}")

    image_name = instance["instance_image"]
    return f"docker.io/bamchil/{image_name}".lower()


def str2bool(v: str) -> bool:
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Boolean value expected, got: {v}")

