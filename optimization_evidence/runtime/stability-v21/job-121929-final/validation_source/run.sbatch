#!/bin/bash
#SBATCH --job-name=arm-stability-v21
#SBATCH --partition=zhangyuelab
#SBATCH --nodelist=nv-h100-022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=none
#SBATCH --time=14:00:00
#SBATCH --output=/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh/papers/active-reward-modeling/acquisition-paper-aligned/stability-v21/job-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp
export ARM_EXPECTED_PHYSICAL_CORES=8
ARM_WORKSPACE=/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh
ARM_PROJECT="$ARM_WORKSPACE/papers/active-reward-modeling/acquisition-paper-aligned"
ARM_RUNTIME="$ARM_WORKSPACE/tmp/harborbuilder-kvm-test"
"$ARM_RUNTIME/engine-unit-venv/bin/python" -B - > "$ARM_PROJECT/stability-v21/runtime-$SLURM_JOB_ID.log" 2>&1 <<'PY' &
import json
import os
from pathlib import Path
import runpy
import sys

root = Path('/storage/zhangyueLab/wengyixuan/ssh_gpu/workspaces/lzh')
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
    identity = ((topology / 'physical_package_id').read_text().strip(),
                (topology / 'core_id').read_text().strip())
    cores.setdefault(identity, cpu)
assert len(cores) >= 8, 'The 8-vCPU VM requires 8 distinct physical host cores'
selected = list(cores.values())[:8]
os.sched_setaffinity(0, selected)
print(json.dumps({'allocated_logical_cpus': int(os.environ['SLURM_CPUS_PER_TASK']),
                  'allocated_physical_cores': len(cores), 'physical_cpus_for_vm': selected,
                  'task_cpus_per_evaluation': 8}), flush=True)
sys.argv = [str(root / 'tmp/harborbuilder-kvm-test/run_vm_service.py'),
            '--cpus', '8', '--memory-mib', '40960', '--disk-gib', '96', '--paper-sandbox',
            '--docker-api-port', '23774',
            '--reuse-data-disk', str(root / 'papers/active-reward-modeling/acquisition-paper-aligned/cold-score-v20/retry-runtime.raw'),
            '--shared-root', str(root)]
runpy.run_path(sys.argv[0], run_name='__main__')
PY
ARM_RUNTIME_PID=$!
trap 'kill "$ARM_RUNTIME_PID" 2>/dev/null || true; wait "$ARM_RUNTIME_PID" 2>/dev/null || true' EXIT
export ARM_RUNTIME_PID
exec_status=0
"$ARM_WORKSPACE/harborbuilder/.venv-harborbuilder/bin/python" -u -B \
  "$ARM_PROJECT/stability-v21/run_stability.py" || exec_status=$?
exit "$exec_status"
