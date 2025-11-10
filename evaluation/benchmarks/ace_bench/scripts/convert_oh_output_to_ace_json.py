#!/usr/bin/env python3
"""
Convert OpenHands output format to ACE-Bench evaluation format.

OpenHands format (output.jsonl):
{
    "instance_id": "...",
    "test_result": {"git_patch": "..."},
    "metadata": {...},
    ...
}

ACE-Bench format (output.ace.jsonl):
{
    "instance_id": "...",
    "model_patch": "...",
    "model_name_or_path": "..."
}
"""

import argparse
import json
from pathlib import Path


def convert_oh_to_ace(oh_output_path: str | Path) -> str:
    """
    Convert OpenHands output to ACE-Bench format.

    Args:
        oh_output_path: Path to OpenHands output.jsonl file

    Returns:
        Path to converted ACE-Bench format file
    """
    oh_output_path = Path(oh_output_path)

    if not oh_output_path.exists():
        raise FileNotFoundError(f"Input file not found: {oh_output_path}")

    # Read OpenHands output
    predictions = []
    with open(oh_output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)

                # Extract required fields
                instance_id = data.get("instance_id")
                if not instance_id:
                    print(f"Warning: Skipping entry without instance_id")
                    continue

                # Extract git patch from test_result
                git_patch = ""
                if "test_result" in data and isinstance(data["test_result"], dict):
                    git_patch = data["test_result"].get("git_patch", "")

                # 确保 patch 以换行符结尾（git 格式规范）
                if git_patch and not git_patch.endswith("\n"):
                    git_patch += "\n"

                # Extract model name from metadata
                model_name = "unknown"
                if "metadata" in data and isinstance(data["metadata"], dict):
                    # Try to get model from llm_config
                    if "llm_config" in data["metadata"]:
                        llm_config = data["metadata"]["llm_config"]
                        if isinstance(llm_config, dict):
                            model_name = llm_config.get("model", "unknown")

                    # Or from agent_class
                    if model_name == "unknown":
                        agent_class = data["metadata"].get("agent_class", "unknown")
                        model_name = agent_class

                # Create ACE-Bench format entry
                ace_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name,
                }

                predictions.append(ace_entry)

    # Write ACE-Bench format output
    output_path = oh_output_path.parent / oh_output_path.name.replace(".jsonl", ".ace.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Converted {len(predictions)} predictions")
    print(f"Output saved to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to ACE-Bench evaluation format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to OpenHands output.jsonl file",
    )

    args = parser.parse_args()

    try:
        convert_oh_to_ace(args.input_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

