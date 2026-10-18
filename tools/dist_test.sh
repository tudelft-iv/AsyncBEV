#!/usr/bin/env bash
# Adapted from the OpenMMLab/CMT distributed launchers.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [evaluation options...]" >&2
    exit 2
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PYTHON=${PYTHON:-python}
if [[ ! $GPUS =~ ^[1-9][0-9]*$ || ! $NNODES =~ ^[1-9][0-9]*$ || ! $NODE_RANK =~ ^[0-9]+$ ]]; then
    echo "GPUS and NNODES must be positive integers; NODE_RANK must be nonnegative." >&2
    exit 2
fi
if (( 10#$NODE_RANK >= NNODES )); then
    echo "NODE_RANK must be smaller than NNODES." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

# The same entrypoint handles ordinary and --eval-dynamic evaluation.
exec "$PYTHON" -m torch.distributed.launch \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$SCRIPT_DIR/test.py" \
    "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"
