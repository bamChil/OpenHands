#!/bin/bash
# ACE-Bench Inference Script
# Based on SWE-Bench's run_infer.sh

set -eo pipefail

# Default values
MODEL_CONFIG=${1:-"llm.eval"}
GIT_VERSION=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-"0"}
MAX_ITER=${5:-"100"}
NUM_WORKERS=${6:-"1"}
DATASET=${7:-"your-org/ACE-bench"}
SPLIT=${8:-"test"}

echo "===================================================="
echo "ACE-Bench Inference Configuration"
echo "===================================================="
echo "Model Config: $MODEL_CONFIG"
echo "Git Version: $GIT_VERSION"
echo "Agent: $AGENT"
echo "Eval Limit: $EVAL_LIMIT"
echo "Max Iterations: $MAX_ITER"
echo "Num Workers: $NUM_WORKERS"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "===================================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ACE_BENCH_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_DIR="$(dirname "$(dirname "$ACE_BENCH_DIR")")"
OPENHANDS_DIR="$(dirname "$EVAL_DIR")"

echo "OpenHands Directory: $OPENHANDS_DIR"
echo "ACE-Bench Directory: $ACE_BENCH_DIR"

# Check if running in development mode
if [ ! -d "$OPENHANDS_DIR/.git" ]; then
    echo "Warning: Not in a git repository. Using current code."
fi

# Set environment variables
export PYTHONPATH="$OPENHANDS_DIR:$PYTHONPATH"

# TODO: 设置Docker镜像前缀
# export EVAL_DOCKER_IMAGE_PREFIX="docker.io/your-org/"

# Check if evaluation output directory should be specified
if [ -z "$EVAL_OUTPUT_DIR" ]; then
    EVAL_OUTPUT_DIR="$OPENHANDS_DIR/evaluation/evaluation_outputs/outputs"
fi

echo "Output Directory: $EVAL_OUTPUT_DIR"

# Build command
CMD="python $ACE_BENCH_DIR/run_infer.py \
    --llm-config $MODEL_CONFIG \
    --agent-cls $AGENT \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --dataset $DATASET \
    --eval-output-dir $EVAL_OUTPUT_DIR"

# Add eval limit if specified
if [ "$EVAL_LIMIT" != "0" ]; then
    CMD="$CMD --eval-n-limit $EVAL_LIMIT"
fi

# Add eval note if specified
if [ -n "$EVAL_NOTE" ]; then
    CMD="$CMD --eval-note \"$EVAL_NOTE\""
fi

echo "===================================================="
echo "Running command:"
echo "$CMD"
echo "===================================================="

# Run the evaluation
eval $CMD

echo "===================================================="
echo "ACE-Bench Inference Complete"
echo "===================================================="

