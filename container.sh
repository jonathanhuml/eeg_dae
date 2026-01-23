#!/bin/bash
set -euo pipefail

NAME="eeg_dae"
IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"

REPO="/data/home/jonhuml/eeg_dae"
DATASETS="/data/datasets/bci"   # BCI subset (matches your earlier setup)

# Must be in an srun step (for proper GPU binding vars)
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
  echo "ERROR: In SLURM job ${SLURM_JOB_ID} but not in an srun step."
  echo "Run: srun --jobid=${SLURM_JOB_ID} --pty bash -l"
  exit 1
fi

# Derive CUDA_VISIBLE_DEVICES from SLURM step if needed
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_STEP_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(echo "${SLURM_STEP_GPUS}" | sed 's/gpu://g' | tr -d ' ')"
fi

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is empty. Allocate a GPU with salloc/srun first.}"

echo "[container.sh] HOST=$(hostname)"
echo "[container.sh] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Recreate container each run to avoid stale mounts
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -t -d \
  --name="$NAME" \
  --network=host --ipc=host \
  --security-opt=no-new-privileges \
  --shm-size=64g \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -v "$REPO":/workspace \
  -v "$DATASETS":/workspace/datasets \
  -w /workspace \
  "$IMAGE" bash

docker exec -it "$NAME" bash
