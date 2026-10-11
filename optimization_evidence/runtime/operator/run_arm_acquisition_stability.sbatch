#!/bin/bash
#SBATCH --job-name=arm-acquisition-stability
#SBATCH --partition=zhangyuelab
#SBATCH --nodelist=nv-h100-022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=none
#SBATCH --time=13:00:00
#SBATCH --output=/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/services/arm-acquisition-stability-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
exec /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/harborbuilder/.venv-harborbuilder/bin/python /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/run_arm_acquisition_stability.py "$@"
