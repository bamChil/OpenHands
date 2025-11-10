#!/usr/bin/env bash

# ACE-Bench Evaluation Script
#
# Usage: ./eval_infer.sh <output_file> [instance_id]
#
# Environment Variables:
#   GPU_IDS: Comma-separated GPU IDs to use (e.g., "2,3"). Default: use all GPUs
#   REVIEW_CODES: Whether to save agent code for review (true/false). Default: false
#
# This script evaluates ACE-Bench predictions by:
# 1. Converting OpenHands output format to ACE-Bench format (if needed)
# 2. Running evaluation using ace_bench.harness.run_evaluation
# 3. Organizing results into eval_outputs directory

PROCESS_FILEPATH=$1
if [ -z "$PROCESS_FILEPATH" ]; then
    echo "Error: PROCESS_FILEPATH is empty. Usage: ./eval_infer.sh <output_file> [instance_id]"
    echo "Environment Variables:"
    echo "  GPU_IDS: Comma-separated GPU IDs to use (e.g., '2,3'). Default: use all GPUs"
    exit 1
fi

if [ ! -f $PROCESS_FILEPATH ]; then
    echo "Error: $PROCESS_FILEPATH is not a file"
    exit 1
fi

# Optional instance_id for single instance evaluation
INSTANCE_ID=$2

echo "Evaluating $PROCESS_FILEPATH"
if [ ! -z "$INSTANCE_ID" ]; then
    echo "Instance ID: $INSTANCE_ID"
fi

PROCESS_FILEPATH=$(realpath $PROCESS_FILEPATH)
FILE_DIR=$(dirname $PROCESS_FILEPATH)
FILE_NAME=$(basename $PROCESS_FILEPATH)

echo "File directory: $FILE_DIR"
echo "File name: $FILE_NAME"

# ================================================
# Detect whether PROCESS_FILEPATH is in ACE-Bench format
echo "=============================================================="
echo "Detecting file format"
echo "=============================================================="

# ACE-Bench format is a JSONL where every line has: instance_id, model_name_or_path, and model_patch
function is_ace_format() {
    # Read the first line of the file
    read -r first_line < "$PROCESS_FILEPATH"

    # Use jq to check if the first line has the required fields
    echo "$first_line" | jq -e '. | has("model_name_or_path") and has("instance_id") and has("model_patch")' > /dev/null

    if [ $? -ne 0 ]; then
        return 1 # Return 1 if the first line does not have the required fields
    fi

    return 0 # Return 0 if the first line has the required fields
}

# Call the function with the file path
is_ace_format "$PROCESS_FILEPATH"
IS_ACE_FORMAT=$?

# Use the result in an if-else statement
if [ $IS_ACE_FORMAT -eq 0 ]; then
    echo "The file IS in ACE-Bench format."
    ACE_FORMAT_JSONL=$PROCESS_FILEPATH
else
    echo "The file IS NOT in ACE-Bench format."

    # ==== Convert OpenHands format to ACE-Bench format ====
    echo "Converting OpenHands output to ACE-Bench format..."
    poetry run python3 evaluation/benchmarks/ace_bench/scripts/convert_oh_output_to_ace_json.py $PROCESS_FILEPATH

    # replace .jsonl with .ace.jsonl in filename
    ACE_FORMAT_JSONL=${PROCESS_FILEPATH/.jsonl/.ace.jsonl}
    echo "ACE_FORMAT_JSONL: $ACE_FORMAT_JSONL"

    # assert that the file exists
    if [ ! -f $ACE_FORMAT_JSONL ]; then
        echo "Error: $ACE_FORMAT_JSONL does not exist. There is probably an error in the conversion process."
        exit 1
    fi
    ACE_FORMAT_JSONL=$(realpath $ACE_FORMAT_JSONL)
fi
# ================================================

echo "=============================================================="
echo "Running ACE-Bench evaluation"
echo "=============================================================="

N_PROCESS=4

# Prepare GPU IDs argument
GPU_ARGS=""
if [ ! -z "$GPU_IDS" ]; then
    GPU_ARGS="--gpu_ids $GPU_IDS"
    echo "Using specific GPUs: $GPU_IDS"
else
    echo "Using all available GPUs"
fi

# Prepare review codes argument
REVIEW_CODES_ARG=""
if [ ! -z "$REVIEW_CODES" ] && [ "$REVIEW_CODES" = "true" ]; then
    REVIEW_CODES_ARG="--review-codes true"
    echo "Review codes: ENABLED"
else
    echo "Review codes: DISABLED"
fi

if [ -z "$INSTANCE_ID" ]; then
    echo "Running ACE-Bench evaluation on the whole input file..."

    poetry run python -m evaluation.benchmarks.ace_bench.harness.run_evaluation \
        --predictions_path $ACE_FORMAT_JSONL \
        --timeout 3600 \
        --max_workers $N_PROCESS \
        $GPU_ARGS \
        $REVIEW_CODES_ARG

    RESULT_OUTPUT_DIR=$(dirname $ACE_FORMAT_JSONL)
    echo "Results saved to $RESULT_OUTPUT_DIR/eval_outputs/"
    echo "Summary report saved to $RESULT_OUTPUT_DIR/report.json"

else
    echo "Running ACE-Bench evaluation on the instance_id: $INSTANCE_ID"

    poetry run python -m evaluation.benchmarks.ace_bench.harness.run_evaluation \
        --predictions_path $ACE_FORMAT_JSONL \
        --timeout 3600 \
        --instance_ids $INSTANCE_ID \
        --max_workers 1 \
        $GPU_ARGS \
        $REVIEW_CODES_ARG

    RESULT_OUTPUT_DIR=$(dirname $ACE_FORMAT_JSONL)
    echo "Evaluation completed for instance $INSTANCE_ID"
    echo "Results saved to $RESULT_OUTPUT_DIR/eval_outputs/$INSTANCE_ID/"
fi

echo "=============================================================="
echo "ACE-Bench evaluation complete!"
echo "=============================================================="

