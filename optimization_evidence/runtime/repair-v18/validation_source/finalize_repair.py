"""Collect actual validation evidence without rewriting successful experiments."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'repair-v18'
D=ROOT/'delivery'
T=D/'workspace/harbor_task'
E=D/'optimization_evidence/runtime/repair-v18'
TRIAL=HERE/'trials/arm-v18-nop-04'
result=json.loads((TRIAL/'result.json').read_text())
assert result['exception_info'] is None
assert result['verifier_environment_mode']=='separate'
assert result['verifier_result']['rewards']=={'score':0.0}
score=json.loads((TRIAL/'verifier/public-score.json').read_text())
assert score['evaluation_split']=='hidden' and score['new_training_runs']==0
hidden=json.loads((E/'hidden-results/comparison.json').read_text())
assert score['baseline_mean']==hidden['baseline_mean']
assert score['raw_values']==hidden['baseline_values']
assert (TRIAL/'verifier/test-stdout.txt').stat().st_size>0
binding=json.loads((HERE/'trial-task-binding.json').read_text())
for name,expected in binding['files'].items():
 assert hashlib.sha256((T/name).read_bytes()).hexdigest()==expected, name
E.mkdir(exist_ok=True)
shutil.copytree(TRIAL,E/'trials'/TRIAL.name,dirs_exist_ok=True)
for name in ['verifier-build-prepare.log','base-load.log','hidden-calibration.log',
             'hidden-runtime-result.log','nop-trial.log','trial-task-binding.json']:
 shutil.copy2(HERE/name,E/name)
for name in ['start_runtime.sbatch','prepare_materials.py','prepare_hidden_runtime.py','run_nop_trial.py','finalize_repair.py']:
 shutil.copy2(HERE/name,E/'validation_source'/name)
# Keep failed and earlier validation records as historical evidence.
for name in ['nop-trial-abstract-api-error.log','nop-trial-constructor-api-error.log',
             'nop-trial-network-policy-error.log','nop-trial-missing-config-error.log','nop-trial-03.log']:
 q=HERE/name
 if q.exists():
  (E/'attempts').mkdir(exist_ok=True)
  shutil.copy2(q,E/'attempts'/name)
for name in ['arm-v18-nop-02','arm-v18-nop-03']:
 q=HERE/'trials'/name
 if q.exists():shutil.copytree(q,E/'attempts'/name,dirs_exist_ok=True)
s=json.loads((D/'expert_evidence/run_summary.json').read_text())
s['status']='implementation_repaired_historical_log_gap'
s['image']='standalone Agent Dev and Hidden verifier builds'
s['current_snapshot'].update({
 'build_passed':True,'full_harbor_trial_passed':True,'hidden_calibration_passed':True,
 'independent_task_contexts':True,'new_training_runs':0,
 'cold_build_passed':False,'fresh_score_within_budget_passed':False,'stability_12h_passed':False,
 'validation_status':'targeted_checks_passed_historical_reference_stdout_missing',
 'trial_evidence':'optimization_evidence/runtime/repair-v18/trials/arm-v18-nop-04/result.json',
 'binding_evidence':'optimization_evidence/runtime/repair-v18/trial-task-binding.json',
 'public_dockerfile_sha256':binding['files']['environment/Dockerfile'],
 'public_grader_sha256':hashlib.sha256((T/'environment/task/tests/grader.py').read_bytes()).hexdigest(),
 'verifier_dockerfile_sha256':binding['files']['tests/Dockerfile'],
 'instruction_sha256':hashlib.sha256((T/'instruction.md').read_bytes()).hexdigest()})
s['validation_scope']='Version 1.2.0 standalone contexts and full official Harbor NOP Trial passed using the bundled StaticOfflineDocker adapter; 24 contract tests and five-seed Hidden Baseline score 0 passed, with zero model API requests and zero new training. B/R Hidden calibration reused all ten preserved checkpoints. Original Reference stdout remains missing. The v14 cold five-seed training and 12h stability records are historical; no new cold-training or 12h acceptance is claimed.'
(D/'expert_evidence/run_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
summary={'task_version':'1.2.0','status':'implementation_repaired',
 'validation':{'full_harbor_nop_trial':'passed','contract_tests':24,
               'hidden_baseline_score':0.0,'hidden_reference_score':hidden['reference_score'],
               'hidden_gain_in_baseline_std':hidden['improvement_in_baseline_std'],
               'new_training_runs':0,'model_api_requests':0,'cpu_allocation':8,'gpu_allocation':0},
 'preserved_successful_models':10,
 'remaining_evidence_gap':'Reference original stdout not recovered; empty historical train.log preserved.',
 'current_version_12h_stability':'not_run','current_version_fresh_training_budget':'not_run',
 'formal_platform_acceptance':False}
(E/'repair-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
