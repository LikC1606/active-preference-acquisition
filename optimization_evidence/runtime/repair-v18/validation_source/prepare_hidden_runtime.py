"""Use this allocation's Docker to evaluate the ten existing B/R checkpoints."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'repair-v18'
DOCKER = [sys.executable,'-B',str(ROOT/'compliance-v14/docker_vm.py')]
NAME='arm-hidden-prepare-v18'

def run(*args, log=None):
    if log:
        with (HERE/log).open('wb') as f:
            return subprocess.run([*DOCKER,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    return subprocess.run([*DOCKER,*args],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)

reference=HERE/'reference-input'
for seed in [101,202,303,404,505]:
    src=ROOT/f'delivery/optimization_evidence/reference_runs/runs/seed_{seed}'
    dst=reference/f'seed_{seed}'
    dst.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src/'model/checkpoint',dst/'checkpoint')
    shutil.copy2(src/'evaluation.json',dst/'evaluation.json')
run('create','--name',NAME,'--network','none','--cpus','8','--memory','36g','--memory-swap','36g',
    '--pids-limit','512','--cap-drop','ALL','--cap-add','SYS_CHROOT','--cap-add','SETUID',
    '--cap-add','SETGID','--cap-add','CHOWN','--cap-add','DAC_OVERRIDE','--cap-add','FOWNER',
    '--security-opt','no-new-privileges','arm-hidden-prepare-v18')
try:
    run('cp',str(reference),NAME+':/tmp/reference')
    run('start',NAME)
    run('exec',NAME,'python3','-u','/opt/arm-private/prepare_hidden_calibration.py',
        '--reference','/tmp/reference','--output','/tmp/hidden-v18',log='hidden-calibration.log')
    run('cp',NAME+':/tmp/hidden-v18',str(HERE/'hidden-results'))
    hidden=HERE/'hidden-results'
    tests=ROOT/'delivery/workspace/harbor_task/tests'
    shutil.copy2(hidden/'baseline-hidden.json',tests/'private/runtime/baseline-hidden.json')
    for seed in [101,202,303,404,505]:
        shutil.copy2(hidden/f'baseline/seed_{seed}/hidden-evaluation.json',
                     tests/f'private/calibration/seed_{seed}/hidden-evaluation.json')
    print(json.dumps({'status':'hidden_calibrated','comparison':json.loads((hidden/'comparison.json').read_text())}))
finally:
    run('rm','--force',NAME)
