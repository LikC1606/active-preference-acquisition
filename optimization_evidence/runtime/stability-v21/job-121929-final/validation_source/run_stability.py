"""Twelve-hour monitoring of the current Dev environment with cached score probes."""
import asyncio
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'cold-score-v20'))
import run_cold_score as cold

cold.HERE = HERE
cold.OUT = HERE / f'job-{cold.JOB}'
OUT = cold.OUT
COLD = PROJECT / 'cold-score-v20/job-121710'
REQUIRED_SECONDS = 43200
DIAGNOSTIC = '''
import json
import shutil
from pathlib import Path
counts = {}
for path in Path('/proc').glob('[0-9]*/fd'):
    try:
        counts[path.parent.name] = len(list(path.iterdir()))
    except (FileNotFoundError, PermissionError):
        continue
channel = Path('/dev/shm/arm-evaluator')
print(json.dumps({'fd_count': max(counts.values(), default=0),
    'total_fd_count': sum(counts.values()), 'process_fd_counts': counts,
    'disk_free_bytes': shutil.disk_usage('/').free,
    'shm_used_bytes': shutil.disk_usage('/dev/shm').used,
    'pending_requests': len(list((channel / 'inbox').glob('*.json'))),
    'completed_responses': len(list((channel / 'outbox').glob('*.json')))}))
'''


def read(path):
    return json.loads(path.read_text())


class NoCacheDocker(cold.StaticOfflineDocker):
    async def _run_docker_compose_command(self, command, **kwargs):
        if command == ['build']:
            command = ['build', '--no-cache']
        return await super()._run_docker_compose_command(command, **kwargs)


def score_probe(container, baseline, expected):
    output = '/harbor_task/solution/artifacts/stability-' + ('baseline' if baseline else 'best')
    command = ['exec', '--workdir', '/harbor_task', container,
               'python3', '/harbor_task/tests/grader.py', '--output', output]
    if baseline:
        command.append('--baseline')
    cold.docker(*command, timeout=90)
    reward = json.loads(cold.docker('exec', container, 'cat', output + '/reward.json'))
    for key in ['status', 'score', 'selector_sha256', 'protocol_sha256',
                'evaluation_split', 'seeds', 'raw_values', 'mean_error']:
        assert reward[key] == expected[key], f'{key} changed during stability scoring'
    assert reward['new_training_runs'] == 0, 'Stability checks must reuse completed models'
    return {'request_id': reward['request_id'], 'score': reward['score'],
            'new_training_runs': reward['new_training_runs']}


async def main():
    cold_result = read(COLD / 'summary.json')
    assert cold_result['status'] == 'passed' and cold_result['old_models_unchanged']
    binding = read(COLD / 'task-binding.json')
    assert cold.digest(cold.SOURCE) == binding['selector_sha256']
    for name, identity in binding['source_files'].items():
        assert cold.digest(cold.TASK / name) == identity, f'Task changed since cold scoring: {name}'
    for name in ['environment/baseline.json', 'environment/dev_runtime/baseline.json',
                 'environment/dev_runtime/definition.json', 'environment/starter/select.py']:
        binding['source_files'][name] = cold.digest(cold.TASK / name)
    for name, identity in binding['preserved_models'].items():
        assert cold.digest(cold.DELIVERY / name) == identity, f'Preserved model changed: {name}'
    expected_best = read(COLD / 'dev/summary.json')['reward']
    expected_baseline = read(cold.TASK / 'environment/baseline.json')
    expected_baseline.update(status='succeeded', evaluation_split='dev')
    protocol = cold.digest(cold.TASK / 'environment/dev_runtime/protocol.json')
    state_path = f'/var/lib/arm/runs/{protocol}/{binding["selector_sha256"]}'
    for seed, metric in zip(cold.SEEDS, expected_best['raw_values']):
        source = COLD / 'dev/runs' / f'seed_{seed}'
        receipt = read(source / 'evaluation.json')
        assert receipt['status'] == 'succeeded' and receipt['raw_metric'] == metric
        assert receipt['checkpoint_reload_match'] is True
        assert cold.digest(source / 'checkpoint') == receipt['checkpoint_sha256']
    binding.update(job_id=cold.JOB, task_version='1.2.0', cold_score_job_id='121710',
                   monitor_sha256=cold.digest(Path(__file__)),
                   instruction_sha256=cold.digest(cold.TASK / 'instruction.md'),
                   public_grader_sha256=cold.digest(cold.TASK / 'environment/task/tests/grader.py'),
                   scope='Public/Dev container pair; Baseline and best cached score probes')
    cold.write(OUT / 'task-binding.json', binding)
    paths = cold.TrialPaths(trial_dir=OUT / 'trial')
    paths.mkdir()
    paths.chmod_dir()
    submitted = OUT / 'artifacts/solution'
    submitted.mkdir(parents=True)
    for name in ['select.py', 'solve.sh', 'submission.json']:
        shutil.copy2(cold.SOURCE if name == 'select.py' else cold.TASK / 'solution' / name,
                     submitted / name)
    task = cold.Task(cold.TASK)
    environment = NoCacheDocker(
        environment_dir=cold.TASK / 'environment', environment_name='arm-stability-v21',
        session_id=f'arm-stability-v21-{cold.JOB}', trial_paths=paths,
        task_env_config=task.config.environment,
        network_policy=cold.NetworkPolicy(network_mode=cold.NetworkMode.NO_NETWORK))
    summary = {'status': 'building', 'job_id': cold.JOB, 'task_version': '1.2.0',
               'required_seconds': REQUIRED_SECONDS, 'scope': binding['scope'],
               'selector_sha256': binding['selector_sha256'], 'protocol_sha256': protocol,
               'cold_score_job_id': '121710', 'monitor_sha256': binding['monitor_sha256'],
               'cycles': 0, 'public_baseline_checks': 0, 'best_method_checks': 0,
               'new_training_runs': 0, 'model_api_requests': 0, 'errors': [],
               'formal_platform_acceptance': False, 'setup_started_at': cold.now()}
    cold.write(OUT / 'summary.json', summary)
    ids = {}
    started = None
    try:
        # Build and start the separate Hidden context before the Dev monitoring window.
        hidden_paths = cold.TrialPaths(trial_dir=OUT / 'hidden-build/trial')
        hidden_paths.mkdir()
        hidden_paths.chmod_dir()
        hidden_environment = NoCacheDocker(
            environment_dir=cold.TASK / 'tests', environment_name='arm-stability-v21-hidden-build',
            session_id=f'arm-stability-v21-hidden-build-{cold.JOB}', trial_paths=hidden_paths,
            task_env_config=task.config.environment,
            network_policy=cold.NetworkPolicy(network_mode=cold.NetworkMode.NO_NETWORK))
        build_record = {'job_id': cold.JOB, 'task_version': '1.2.0', 'status': 'building',
                        'build_cache_disabled': True, 'contexts': {}, 'started_at': cold.now()}
        cold.write(OUT / 'no-cache-build.json', build_record)
        try:
            build_started = time.monotonic()
            await asyncio.wait_for(hidden_environment.start(force_build=True), timeout=1800)
            result = await hidden_environment._run_docker_compose_command(['ps', '-q', 'main'])
            hidden_inspect = json.loads(cold.docker('inspect', result.stdout.strip()))
            assert hidden_inspect[0]['State']['Running']
            assert hidden_inspect[0]['State']['Health']['Status'] == 'healthy'
            cold.write(OUT / 'hidden-build/initial-inspect.json', hidden_inspect)
            build_record['contexts']['hidden'] = {'status': 'passed',
                'elapsed_seconds': time.monotonic() - build_started,
                'image': hidden_inspect[0]['Image'], 'new_training_runs': 0}
            cold.write(OUT / 'no-cache-build.json', build_record)
        finally:
            await hidden_environment.stop(delete=True)
        build_started = time.monotonic()
        await asyncio.wait_for(environment.start(force_build=True), timeout=1800)
        for service in ['main', 'evaluator']:
            result = await environment._run_docker_compose_command(['ps', '-q', service])
            ids[service] = result.stdout.strip()
            assert ids[service], f'Missing {service} container'
        cold.docker('update', '--cpuset-cpus', '0-5', ids['evaluator'])
        cold.docker('update', '--cpuset-cpus', '6-7', ids['main'])
        initial = json.loads(cold.docker('inspect', *ids.values()))
        images = {row['Id']: row['Image'] for row in initial}
        cold.write(OUT / 'initial-inspect.json', initial)
        build_record['contexts']['dev'] = {'status': 'passed',
            'elapsed_seconds': time.monotonic() - build_started, 'images': images,
            'new_training_runs': 0}
        build_record.update(status='passed', finished_at=cold.now())
        cold.write(OUT / 'no-cache-build.json', build_record)
        info = json.loads(cold.docker('info', '--format', '{{json .}}'))
        assert info['MemoryLimit'] and info['CpuCfsQuota']
        swaps = cold.docker('exec', ids['evaluator'], 'cat', '/proc/swaps')
        assert not any(line.strip() for line in swaps.splitlines()[1:]), 'Active VM swap detected'
        cold.docker('exec', ids['evaluator'], 'test', '!', '-e', state_path)
        cold.docker('exec', ids['evaluator'], 'mkdir', '-p', str(Path(state_path).parent))
        cold.docker('cp', COLD / 'dev/runs', ids['evaluator'] + ':' + state_path)
        handler = cold.ArtifactHandler(artifacts=task.config.artifacts, logger=logging.getLogger(__name__))
        env_paths = cold.EnvironmentPaths.for_os(environment.os)
        await handler.upload_artifacts(environment, OUT / 'artifacts',
            source_artifacts_dir=env_paths.artifacts_dir, target_artifacts_dir=env_paths.artifacts_dir)
        actual = await environment.exec('sha256sum solution/select.py', cwd='/harbor_task')
        assert actual.return_code == 0 and actual.stdout.split()[0] == binding['selector_sha256']
        inside = json.loads(cold.docker('exec', ids['evaluator'], 'cat', '/var/lib/arm/runtime.json'))
        assert inside['protocol_sha256'] == protocol
        assert inside['service_sha256'] == binding['source_files']['environment/dev_runtime/service.py']
        assert inside['worker_sha256'] == binding['source_files']['environment/dev_runtime/worker.py']
        cold.write(OUT / 'runtime.json', inside)
        started = time.monotonic()
        summary.update(status='running', started_at=cold.now(), container_ids=ids, images=images,
                       copied_validated_seed_models=cold.SEEDS, swap_accounting=info['SwapLimit'])
        cold.write(OUT / 'summary.json', summary)
        print(json.dumps({'event': 'stability_started', 'job_id': cold.JOB,
                          'required_seconds': REQUIRED_SECONDS, 'scope': summary['scope']}), flush=True)
        with (OUT / 'samples.jsonl').open('x') as samples:
            while True:
                inspected = json.loads(cold.docker('inspect', *ids.values()))
                rows = []
                for row in inspected:
                    state, limits = row['State'], row['HostConfig']
                    assert state['Running'] and not state['OOMKilled'], state
                    assert row['RestartCount'] == 0 and row['Image'] == images[row['Id']]
                    assert not row['Mounts'] and limits['NetworkMode'] == 'none'
                    assert not limits.get('DeviceRequests') and not limits.get('Devices')
                    assert state.get('Health', {}).get('Status', 'healthy') == 'healthy'
                    if info['SwapLimit']:
                        assert limits['MemorySwap'] == limits['Memory']
                    assert limits['LogConfig']['Config'] == {'max-file': '2', 'max-size': '10m'}
                    diagnostic = json.loads(cold.docker('exec', row['Id'], 'python3', '-c', DIAGNOSTIC))
                    assert diagnostic['fd_count'] < 256 and diagnostic['total_fd_count'] < 1024, diagnostic
                    assert diagnostic['disk_free_bytes'] > 2 * 1024**3, diagnostic
                    assert diagnostic['shm_used_bytes'] < 48 * 1024**2, diagnostic
                    assert diagnostic['pending_requests'] == 0, diagnostic
                    # The trusted service retains one response per completed score request.
                    assert diagnostic['completed_responses'] <= 2 * summary['cycles'], diagnostic
                    rows.append({'id': row['Id'], 'image': row['Image'], 'state': state,
                                 'nano_cpus': limits['NanoCpus'], 'memory': limits['Memory'],
                                 'cpuset_cpus': limits['CpusetCpus'], 'logging': limits['LogConfig'],
                                 **diagnostic})
                assert sum(row['nano_cpus'] for row in rows) == 8 * 10**9
                assert sum(row['memory'] for row in rows) == 36 * 1024**3
                baseline = score_probe(ids['main'], True, expected_baseline)
                best = score_probe(ids['main'], False, expected_best)
                stats = [json.loads(line) for line in cold.docker(
                    'stats', '--no-stream', '--format', '{{json .}}', *ids.values()).splitlines()]
                elapsed = time.monotonic() - started
                samples.write(json.dumps({'at': cold.now(), 'elapsed_seconds': elapsed,
                    'containers': rows, 'stats': stats, 'baseline': baseline, 'best_method': best}) + '\n')
                samples.flush()
                summary.update(cycles=summary['cycles'] + 1,
                               public_baseline_checks=summary['public_baseline_checks'] + 1,
                               best_method_checks=summary['best_method_checks'] + 1,
                               elapsed_seconds=elapsed, last_sample_at=cold.now())
                cold.write(OUT / 'summary.json', summary)
                if elapsed >= REQUIRED_SECONDS:
                    break
                await asyncio.sleep(min(60, REQUIRED_SECONDS - elapsed))
        for name, identity in binding['source_files'].items():
            assert cold.digest(cold.TASK / name) == identity, f'Task changed during stability: {name}'
        assert cold.digest(cold.TASK / 'instruction.md') == binding['instruction_sha256']
        assert cold.digest(cold.TASK / 'environment/task/tests/grader.py') == binding['public_grader_sha256']
        assert cold.digest(cold.SOURCE) == binding['selector_sha256']
        for name, identity in binding['preserved_models'].items():
            assert cold.digest(cold.DELIVERY / name) == identity, f'Preserved model changed: {name}'
        summary.update(status='passed', old_models_unchanged=True, task_files_unchanged=True)
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__,
                       traceback=traceback.format_exc(limit=8)[-4500:])
        summary['errors'].append((str(error) or repr(error))[-3500:])
    finally:
        if ids:
            try:
                cold.write(OUT / 'final-inspect.json', json.loads(cold.docker('inspect', *ids.values())))
                for service, container in ids.items():
                    cold.docker('logs', '--tail', '1000', container, output=OUT / f'{service}-service.log')
            except Exception as error:
                summary['diagnostic_export_error'] = repr(error)[-2000:]
        try:
            await environment.stop(delete=True)
        except Exception as error:
            summary['cleanup_error'] = repr(error)[-2000:]
            summary['status'] = 'failed'
            summary['errors'].append(summary['cleanup_error'])
        summary['finished_at'] = cold.now()
        if started is not None:
            summary['elapsed_seconds'] = time.monotonic() - started
        cold.write(OUT / 'summary.json', summary)
        print(json.dumps({'event': 'stability_finished', 'job_id': cold.JOB,
                          'status': summary['status'], 'elapsed_seconds': summary.get('elapsed_seconds'),
                          'errors': summary['errors']}), flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    try:
        cold.setup()
        logging.basicConfig(filename=OUT / 'operator.log', level=logging.INFO)
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        cold.write(OUT / 'setup-failure.json', {'status': 'failed', 'job_id': cold.JOB,
            'error_type': type(error).__name__, 'error': (str(error) or repr(error))[-3500:],
            'traceback': traceback.format_exc(limit=8)[-4500:], 'finished_at': cold.now()})
        raise
