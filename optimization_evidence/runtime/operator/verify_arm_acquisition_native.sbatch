#!/bin/bash
#SBATCH --job-name=arm-acquisition-native
#SBATCH --partition=zhangyuelab
#SBATCH --nodelist=nv-h100-018
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --output=/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/services/arm-acquisition-native-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
exec /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/harborbuilder/.venv-harborbuilder/bin/python /storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/tmp/harborbuilder-kvm-test/verify_arm_acquisition_native.py "$@"
