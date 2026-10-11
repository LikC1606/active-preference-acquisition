"""Run an unmodified Harbor Trial for the repaired task, using no model API."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'repair-v18'
RUNTIME=ROOT.parents[2]/'tmp/harborbuilder-kvm-test'
status=json.loads((RUNTIME/f"services/job-{os.environ['SLURM_JOB_ID']}/status.json").read_text())
assert status['status']=='running' and status['node']==socket.gethostname()
assert status['vcpus']<=8 and status['gpu_count']==0
os.environ['PATH']=str(RUNTIME/'deployment')+os.pathsep+os.environ['PATH']
os.environ['DOCKER_HOST']=f"tcp://127.0.0.1:{status['docker_api_port']}"
os.environ['DOCKER_TLS_VERIFY']='1'
tls=HERE/'docker-client-tls'
tls.mkdir(exist_ok=True)
for dst,src in [('ca.pem','ca.pem'),('cert.pem','client-cert.pem'),('key.pem','client-key.pem')]:
    shutil.copy2(RUNTIME/'deployment/tls'/src,tls/dst)
    (tls/dst).chmod(0o600)
os.environ['DOCKER_CERT_PATH']=str(tls)
sys.path.insert(0, str(ROOT/'delivery/workspace/harbor_task/environment'))
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

TASK=ROOT/'delivery/workspace/harbor_task'
config=TrialConfig.model_validate({'task':{'path':str(TASK)},'trial_name':'arm-v18-nop-04',
    'trials_dir':str(HERE/'trials'),'agent':{'name':'nop'},
    'environment':{'import_path':'arm_static_offline:StaticOfflineDocker','delete':True},
    'verifier':{'disable':False}})
async def main():
    trial=await Trial.create(config)
    result=await trial.run()
    record=result.model_dump(mode='json')
    binding={'task_version':'1.2.0','job_id':os.environ['SLURM_JOB_ID'],
             'vcpus':status['vcpus'],'gpu_count':0,
             'files':{p:hashlib.sha256((TASK/p).read_bytes()).hexdigest() for p in [
                 'task.toml','environment/Dockerfile','environment/Dev.Dockerfile',
                 'environment/docker-compose.yaml','environment/arm_static_offline.py',
                 'environment/dev_runtime/service.py','environment/dev_runtime/worker.py',
                 'tests/Dockerfile','tests/docker-compose.yaml','tests/test.sh',
                 'tests/private/runtime/service.py','tests/private/runtime/worker.py',
                 'tests/private/runtime/validate_calibration.py','tests/private/runtime/baseline-hidden.json',
                 'tests/hidden_assets/manifest.json']}}
    (HERE/'trial-task-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps({'exception_info':record.get('exception_info'),
                     'verifier_result':record.get('verifier_result'),'finished_at':record.get('finished_at')}))
    assert record.get('exception_info') is None,record.get('exception_info')
    assert record['verifier_result']['rewards']=={'score':0.0},record['verifier_result']
asyncio.run(main())
