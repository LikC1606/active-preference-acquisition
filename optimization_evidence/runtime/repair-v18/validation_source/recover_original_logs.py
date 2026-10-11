"""Copy original Docker json-file records, never synthesize historical output."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent
D=ROOT/'delivery'
E=D/'optimization_evidence/runtime/repair-v18'
index=json.loads((ROOT/'repair-v18/old-disk-container-index.json').read_text())
item=next(r for r in index if r['name']=='/arm-native-v14-reference-evaluator-1')
source=ROOT/'repair-v18/recovered-docker'/(item['id']+'.json.log')
original=source.read_bytes()
reference_sha=hashlib.sha256((D/'workspace/reference/select.py').read_bytes()).hexdigest()
by_seed={}
for line_number,line in enumerate(original.splitlines(keepends=True),1):
 envelope=json.loads(line)
 try:payload=json.loads(envelope['log'])
 except ValueError:continue
 if not isinstance(payload,dict) or payload.get('selector_sha256')!=reference_sha or 'seed' not in payload:continue
 seed=payload['seed']
 receipt=json.loads((D/f'optimization_evidence/reference_runs/runs/seed_{seed}/evaluation.json').read_text())
 assert payload['raw_metric']==receipt['raw_metric']
 by_seed.setdefault(seed,[]).append({'line':line,'line_number':line_number,'time':envelope['time']})
assert set(by_seed)=={101,202,303,404,505}
ref=D/'optimization_evidence/reference_runs'
(ref/'original-evaluator.json.log').write_bytes(original)
provenance={'operation':'read_only_ext4_debugfs_dump','source_disk':'tmp/harborbuilder-kvm-test/services/job-114749/data.raw',
 'source_log_path':item['log_path'],'container_id':item['id'],'container_name':item['name'],
 'container_created':item['created'],'container_image':item['image'],
 'source_log_sha256':hashlib.sha256(original).hexdigest(),'source_log_bytes':len(original),
 'selector_sha256':reference_sha,'role':'unaltered historical evaluator stdout/stderr with Docker timestamps',
 'worker_training_stdout_emitted':False,
 'note':'These are real evaluator execution records, not detailed optimizer stdout. Training history/accounting and model receipts remain separate and unchanged.',
 'seeds':{str(seed):{'source_lines':[r['line_number'] for r in rows],'times':[r['time'] for r in rows]} for seed,rows in by_seed.items()}}
(ref/'log-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for seed,rows in by_seed.items():
 (ref/f'runs/seed_{seed}/run.log').write_bytes(b''.join(row['line'] for row in rows))
# Baseline's pre-existing plain stdout has one original line per seed.
base=D/'optimization_evidence/baseline_runs'
for line in (base/'original-evaluator.log').read_bytes().splitlines(keepends=True):
 d=json.loads(line)
 if 'seed' in d:
  (base/f"runs/seed_{d['seed']}/run.log").write_bytes(line)
shutil.copy2(ROOT/'repair-v18/recover_original_logs.py',E/'validation_source/recover_original_logs.py')
s=json.loads((D/'expert_evidence/run_summary.json').read_text())
s['status']='implementation_repaired_original_logs_recovered'
s['current_snapshot']['validation_status']='targeted_checks_passed_original_evaluator_logs_recovered'
s['validation_scope']=s['validation_scope'].replace('Original Reference stdout remains missing.',
 'Original Baseline evaluator stdout and Reference Docker json-file execution records have been recovered and mapped to all five seeds. Worker train.log remains empty because the historical trainer was silent.')
(D/'expert_evidence/run_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
r=json.loads((E/'repair-summary.json').read_text())
r.pop('remaining_evidence_gap',None)
r['original_execution_logs']='Recovered Baseline stdout and Reference Docker json-file records; all five seeds match formal metrics.'
r['worker_stdout_note']='Historical train.log remains empty; genuine evaluator logs supplied as run.log with provenance. Future worker emits actual round events.'
(E/'repair-summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'original_execution_logs_recovered','reference_seeds':sorted(by_seed),
                  'reference_original_bytes':len(original),'new_training_runs':0}))
