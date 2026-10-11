#!/bin/bash
#SBATCH --job-name=arm-qa-repair-v18
#SBATCH --partition=zhangyuelab
#SBATCH --nodelist=nv-h100-022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=none
#SBATCH --time=02:00:00
#SBATCH --output=/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/papers/active-reward-modeling/acquisition-paper-aligned/repair-v18/runtime-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp
exec /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/engine-unit-venv/bin/python -B \
 /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/run_vm_service.py \
 --cpus 8 --memory-mib 40960 --disk-gib 96 --paper-sandbox --docker-api-port 23771 \
 --shared-root /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh
