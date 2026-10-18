#!/usr/bin/env bash
# Adapted from the OpenMMLab/CMT distributed launchers.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [training options...]" >&2
    exit 2
fi

CONFIG=$1
GPUS=$2
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

# Keep the caller's CUDA_VISIBLE_DEVICES and scheduler/resource settings.
# distributed.launch supports the documented PyTorch 1.9 / 1.11 environments.
exec "$PYTHON" -m torch.distributed.launch \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$SCRIPT_DIR/train.py" \
    "$CONFIG" --launcher pytorch "${@:3}"
