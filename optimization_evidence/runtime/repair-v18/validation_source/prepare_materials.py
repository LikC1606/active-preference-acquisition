"""Prepare independent verifier inputs from existing, unchanged local artifacts."""
import hashlib
import json
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'delivery'
T = D / 'workspace/harbor_task'
N = D / 'optimization_evidence/runtime/native'
S = ROOT.parent / 'data/harmless-gpt4-gemma7b-gemma2b/quickstart-split'
V = T / 'tests'

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def copy(src, dst, large=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return
        dst.unlink()
    if large:
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)

receipts = {split: json.loads((S / name).read_text()) for split,name in [
    ('train','receipt.json'), ('dev','dev-receipt.json'), ('hidden','hidden-receipt.json')]}
ids = {split:set(r['source_prompt_indices']) for split,r in receipts.items()}
assert [len(ids[s]) for s in ['train','dev','hidden']] == [33500,4188,4188]
assert not (ids['train'] & ids['dev'] or ids['train'] & ids['hidden'] or ids['dev'] & ids['hidden'])
assert len(ids['train'] | ids['dev'] | ids['hidden']) == 41876
assert len({r['split_plan_sha256'] for r in receipts.values()}) == 1
# Bind existing train/dev runtime files to the same split receipt as Hidden.
for split, folder in [('train',T/'environment/public_assets'),('dev',T/'environment/public_assets')]:
    source_file = next(f for f in receipts[split]['files'] if 'embeddings' in f['file'])
    public_manifest = json.loads((folder/'data_manifest.json').read_text())
    observed = next(f for f in public_manifest['files'] if f['file']==source_file['file'])
    assert source_file['sha256']==observed['sha256']
for item in receipts['hidden']['files']:
    src=S/item['file']
    assert src.stat().st_size == item['bytes']
    with src.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    copy(src,V/'hidden_assets'/item['file'],True)
for split,name in [('train','receipt.json'),('dev','dev-receipt.json'),('hidden','hidden-receipt.json')]:
    copy(S/name,V/'hidden_assets'/name)
manifest={'schema':'arm.hidden-data.v1','split':'hidden','prompt_count':4188,
          'split_plan_sha256':receipts['hidden']['split_plan_sha256'],
          'files':receipts['hidden']['files'], 'disjoint_from_train_and_dev':True,
          'source':'held-out prompts from the original author quickstart training data',
          'official_paper_test_split':False,
          'receipts':{s:hashlib.sha256((S/n).read_bytes()).hexdigest() for s,n in [
              ('train','receipt.json'),('dev','dev-receipt.json'),('hidden','hidden-receipt.json')]}}
write(V/'hidden_assets/manifest.json',manifest)
copy(T/'environment/requirements.txt',V/'requirements.txt')
for src in (T/'environment/wheels').iterdir():
    if src.is_file():copy(src,V/'wheels'/src.name,True)
for src in (T/'environment/starter').iterdir():
    if src.is_file():copy(src,V/'starter'/src.name)
copy(T/'environment/task/submission.json',V/'submission.json')
for src in (T/'environment/public_assets').iterdir():
    if src.is_file():copy(src,V/'private/public_assets'/src.name,src.suffix=='.npy')
for src in (N/'paper').rglob('*'):
    if src.is_file():copy(src,V/'private/paper'/src.relative_to(N/'paper'),src.suffix=='.npy')
for src in N.glob('*.py'):copy(src,V/'private/runtime'/src.name)
for name in ['baseline.json','definition.json']:copy(N/name,V/'private/runtime'/name)
copy(N/'protocol.json',V/'private/protocol.json')
for src in (N/'calibration').rglob('*'):
    if src.is_file():copy(src,V/'private/calibration'/src.relative_to(N/'calibration'))
(V/'docker-compose.yaml').write_text('''services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
      network: none
    image: ${MAIN_IMAGE_NAME:-arm-acquisition-hidden-v18}
    user: "0:0"
    network_mode: none
    shm_size: 64m
    cpus: 8
    mem_limit: 36g
    memswap_limit: 36g
    pids_limit: 512
    init: true
    cap_drop: [ALL]
    cap_add: [SYS_CHROOT, SETUID, SETGID, CHOWN, DAC_OVERRIDE, FOWNER]
    security_opt: [no-new-privileges:true]
    command: ["sleep", "infinity"]
    logging:
      driver: json-file
      options:
        max-size: 10m
        max-file: "2"
''')
# Original attempts remain untouched; the official trajectories are projections.
for lane,name in [('gpt','trajectory_codex.json'),('seed','trajectory_seed.json')]:
    p=D/f'optimization_evidence/runtime/agent-research/{lane}'
    rows=[]
    for f in sorted(p.glob('attempt-*/result.json')):
        d=json.loads(f.read_text())
        description=d.get('method_summary','Model call failed before a candidate was produced.')
        if lane=='gpt' and ('prompt' in description.lower()):
            description+=' Implementation note: candidate rows are shuffled upstream; the 45-row blocks used here are arbitrary candidate blocks and do not guarantee one pair per prompt.'
        rows.append({'round':d['attempt'],'policy_name':f"{d['model']}/attempt-{d['attempt']:03d}",
                     'method_summary':description,'status':d['status'],'score':d.get('score'),
                     'failure_reason':d.get('error') if d['status']!='succeeded' else None,
                     'retained_best':d.get('decision')=='retain','time':d.get('finished_at') or d.get('started_at'),
                     'effective_round':d['round'],'effective_seconds':d.get('effective_seconds',0.0) if d['status']=='succeeded' else 0.0,
                     'started_at':d.get('started_at'),'source_result':str(f.relative_to(D)),
                     'selector_sha256':d.get('selector_sha256')})
    assert all(r['time'] for r in rows)
    write(D/'expert_evidence'/name,{'rounds':rows})
    write(p/'original_trajectory.json',{'rounds':rows})
# Revised runtime needs new validation, not inherited v14 acceptance flags.
s= json.loads((D/'expert_evidence/run_summary.json').read_text())
s['historical_snapshot_v17']=s['current_snapshot']
s['current_snapshot']={'task_version':'1.2.0','cold_build_passed':False,
                       'fresh_score_within_budget_passed':False,'stability_12h_passed':False,
                       'validation_status':'repair_in_progress'}
s['validation_scope']='The v14 build, five-seed training and 12h stability records are historical. Version 1.2.0 adds independent Hidden grading and requires its own validation.'
s['stability']['current_environment_status']='historical_pass_current_version_not_run'
s['stability']['current_environment_passed']=False
write(D/'expert_evidence/run_summary.json',s)
print(json.dumps({'status':'materials_prepared','hidden_prompts':4188,'disjoint':True,
                  'new_training_runs':0,'trajectory_attempts':{'gpt':7,'seed':11}}))
