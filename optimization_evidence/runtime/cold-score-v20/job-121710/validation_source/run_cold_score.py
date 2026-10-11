"""Fresh best-method Dev/Hidden scoring in one CPU-only allocation, without LLMs."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parents[2]
RUNTIME = WORKSPACE / 'tmp/harborbuilder-kvm-test'
DELIVERY = PROJECT / 'delivery'
TASK = DELIVERY / 'workspace/harbor_task'
JOB = os.environ['SLURM_JOB_ID']
OUT = HERE / f'job-{JOB}'
SOURCE = DELIVERY / 'expert_evidence/best_method/select.py'
SEEDS = [101, 202, 303, 404, 505]

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)

def docker(*args, output=None, timeout=600):
    command = [str(RUNTIME / 'deployment/docker'), *map(str, args)]
    if output is not None:
        with output.open('wb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Docker command failed ({result.returncode}); see {output.name}')
        return ''
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
    return result.stdout.decode()

def setup():
    OUT.mkdir(parents=True, exist_ok=False)
    state_path = RUNTIME / f'services/job-{JOB}/status.json'
    expires = time.monotonic() + 300
    while time.monotonic() < expires:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['status'] == 'running':
                break
            if state['status'] in ['failed', 'stopped']:
                raise RuntimeError(f'VM setup failed: {state.get("error", state["status"])}')
        os.kill(int(os.environ['ARM_RUNTIME_PID']), 0)
        time.sleep(2)
    else:
        raise TimeoutError('VM startup did not complete within 300 seconds')
    assert state['node'] == socket.gethostname() and state['vcpus'] == 8 and state['gpu_count'] == 0
    os.environ['PATH'] = str(RUNTIME / 'deployment') + os.pathsep + os.environ['PATH']
    os.environ['DOCKER_HOST'] = f"tcp://127.0.0.1:{state['docker_api_port']}"
    os.environ['DOCKER_TLS_VERIFY'] = '1'
    certs = HERE / 'docker-client-tls'
    certs.mkdir(exist_ok=True)
    for destination, source in [('ca.pem', 'ca.pem'), ('cert.pem', 'client-cert.pem'), ('key.pem', 'client-key.pem')]:
        shutil.copy2(RUNTIME / 'deployment/tls' / source, certs / destination)
        (certs / destination).chmod(0o600)
    os.environ['DOCKER_CERT_PATH'] = str(certs)
    affinity = sorted(os.sched_getaffinity(0))
    physical = set()
    for cpu in affinity:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.add(((base / 'physical_package_id').read_text().strip(), (base / 'core_id').read_text().strip()))
    qemu_affinity = sorted(os.sched_getaffinity(state['qemu_pid']))
    qemu_physical = set()
    for cpu in qemu_affinity:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        qemu_physical.add(((base / 'physical_package_id').read_text().strip(), (base / 'core_id').read_text().strip()))
    expected_physical = int(os.environ.get('ARM_EXPECTED_PHYSICAL_CORES', '0'))
    if expected_physical:
        assert len(physical) >= expected_physical, 'Insufficient physical host cores'
        assert len(qemu_affinity) == len(qemu_physical) == expected_physical, 'VM must use one logical CPU per physical core'
        assert set(qemu_affinity).issubset(affinity), 'VM CPUs must belong to this allocation'
    write(OUT / 'allocation.json', {'job_id': JOB, 'runtime': state,
          'host_cpu_affinity': affinity, 'host_physical_cores': len(physical),
          'allocated_logical_cpus': int(os.environ['SLURM_CPUS_PER_TASK']),
          'qemu_cpu_affinity': qemu_affinity, 'qemu_physical_cores': len(qemu_physical),
          'model_api_requests': 0, 'started_at': now()})
    docker('load', '--input', PROJECT / 'compliance-v14/python-base-offline.tar',
           output=OUT / 'base-load.log', timeout=900)

sys.path.insert(0, str(TASK / 'environment'))
from arm_static_offline import StaticOfflineDocker
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.task.task import Task
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.verifier.verifier import Verifier

async def telemetry(ids, path):
    while True:
        result = await asyncio.create_subprocess_exec(
            str(RUNTIME / 'deployment/docker'), 'stats', '--no-stream', '--format', '{{json .}}',
            *ids, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await result.communicate()
        with path.open('a') as stream:
            stream.write(json.dumps({'time': now(), 'exit_code': result.returncode,
                                     'stats': stdout.decode()[:6000],
                                     'error': stderr.decode()[-1000:]}) + '\n')
        await asyncio.sleep(30)

async def phase(split):
    output = OUT / split
    paths = TrialPaths(trial_dir=output / 'trial')
    paths.mkdir()
    paths.chmod_dir()
    submitted = output / 'artifacts/solution'
    submitted.mkdir(parents=True)
    for name in ['select.py', 'solve.sh', 'submission.json']:
        shutil.copy2(SOURCE if name == 'select.py' else TASK / 'solution' / name, submitted / name)
    identity = digest(SOURCE)
    # The worker checks the immutable protocol file, whose whitespace is part of its identity.
    protocol = digest(TASK / 'environment/dev_runtime/protocol.json')
    state_path = f'/var/lib/arm/runs/{protocol}/{identity}' + ('/hidden' if split == 'hidden' else '')
    task = Task(TASK)
    environment = StaticOfflineDocker(environment_dir=TASK / ('tests' if split == 'hidden' else 'environment'),
        environment_name='arm-cold-v20-' + split, session_id=f'arm-cold-v20-{split}-{JOB}',
        trial_paths=paths, task_env_config=task.config.environment,
        mounts=[{'type': 'bind', 'source': str(paths.verifier_dir.resolve()), 'target': '/logs/verifier'}],
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    record = {'status': 'building', 'split': split, 'task_version': '1.2.0', 'job_id': JOB,
              'selector_sha256': identity, 'protocol_sha256': protocol, 'setup_started_at': now(),
              'cold_candidate_cache': True, 'score_budget_seconds': 7200,
              'model_api_requests': 0, 'formal_platform_acceptance': False}
    write(output / 'summary.json', record)
    ids = {}
    monitor = None
    started = None
    try:
        await environment.start(force_build=True)
        for service in (['main'] if split == 'hidden' else ['main', 'evaluator']):
            result = await environment._run_docker_compose_command(['ps', '-q', service])
            ids[service] = result.stdout.strip()
            assert ids[service]
        if split == 'dev':
            docker('update', '--cpuset-cpus', '0-5', ids['evaluator'])
            docker('update', '--cpuset-cpus', '6-7', ids['main'])
        else:
            docker('update', '--cpuset-cpus', '0-7', ids['main'])
        inspected = json.loads(docker('inspect', *ids.values()))
        assert sum(i['HostConfig']['NanoCpus'] for i in inspected) == 8 * 10**9
        assert sum(i['HostConfig']['Memory'] for i in inspected) == 36 * 1024**3
        assert all(i['HostConfig']['NetworkMode'] == 'none'
                   and not i['HostConfig'].get('DeviceRequests') for i in inspected)
        write(output / 'initial-inspect.json', inspected)
        private_id = ids['main' if split == 'hidden' else 'evaluator']
        info = json.loads(docker('info', '--format', '{{json .}}'))
        assert info['MemoryLimit'] and info['CpuCfsQuota'], 'VM kernel must enforce CPU and memory limits'
        swaps = docker('exec', private_id, 'cat', '/proc/swaps')
        active_swaps = [line for line in swaps.splitlines()[1:] if line.strip()]
        assert not active_swaps, 'Scoring VM has active swap; cold validation requires swap disabled'
        if info['SwapLimit']:
            assert all(i['HostConfig']['MemorySwap'] == i['HostConfig']['Memory'] for i in inspected), \
                'Docker swap accounting is available but container swap limits differ from memory limits'
        write(output / 'resource-check.json', {
            'cpu_cores': 8, 'memory_gib': 36, 'gpu_count': 0, 'network': 'none',
            'docker_swap_accounting': info['SwapLimit'], 'proc_swaps': swaps,
            'active_swap_devices': 0, 'memory_limit_enforced': info['MemoryLimit'],
            'swap_check': 'Docker limits when supported; always require no active swap devices'})
        docker('exec', private_id, 'test', '!', '-e', state_path)
        handler = ArtifactHandler(artifacts=task.config.artifacts, logger=logging.getLogger(__name__))
        env_paths = EnvironmentPaths.for_os(environment.os)
        await handler.upload_artifacts(environment, output / 'artifacts',
            source_artifacts_dir=env_paths.artifacts_dir, target_artifacts_dir=env_paths.artifacts_dir)
        actual = await environment.exec('sha256sum solution/select.py', cwd='/harbor_task')
        assert actual.return_code == 0 and actual.stdout.split()[0] == identity
        monitor = asyncio.create_task(telemetry(list(ids.values()), output / 'telemetry.jsonl'))
        started = time.monotonic()
        record.update(status='running', started_at=now(), cache_absent_before_start=True, container_ids=ids)
        write(output / 'summary.json', record)
        print(json.dumps({'event': 'cold_score_started', 'split': split, 'output': str(output)}), flush=True)
        if split == 'hidden':
            verifier = Verifier(task=task, trial_paths=paths, environment=environment, skip_tests_upload=True)
            result = await asyncio.wait_for(verifier.verify(), timeout=7200)
            reward = json.loads((paths.verifier_dir / 'public-score.json').read_text())
            assert result.rewards == {'score': reward['score']}
        else:
            result = await environment.exec(
                'python3 /harbor_task/tests/grader.py --output solution/artifacts/cold-dev --verifier',
                cwd='/harbor_task', timeout_sec=7200)
            (output / 'grader-stdout.txt').write_text((result.stdout or '')[-16000:])
            (output / 'grader-stderr.txt').write_text((result.stderr or '')[-16000:])
            assert result.return_code == 0, 'Dev grader failed; see grader output'
            reward = json.loads((paths.verifier_dir / 'public-score.json').read_text())
        elapsed = time.monotonic() - started
        assert elapsed <= 7200 and reward['status'] == 'succeeded' and reward['new_training_runs'] == 5
        assert reward['selector_sha256'] == identity and reward['protocol_sha256'] == protocol
        assert reward['seeds'] == SEEDS and reward['evaluation_split'] == split
        assert reward['mean_error'] == math.fsum(reward['raw_values']) / 5
        assert reward['score'] == (reward['baseline_mean'] - reward['mean_error']) / reward['baseline_mean']
        record.update(status='passed', elapsed_seconds=elapsed, reward=reward, within_7200_seconds=True)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__,
                      error=(str(error) or repr(error))[-3500:], traceback=traceback.format_exc(limit=8)[-4500:])
        if started is not None:
            record['elapsed_seconds'] = time.monotonic() - started
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if ids:
            private_id = ids.get('main' if split == 'hidden' else 'evaluator')
            try:
                docker('cp', private_id + ':' + state_path, output / 'runs', timeout=300)
                record['completed_new_training_seeds'] = sorted(
                    int(p.parent.name.removeprefix('seed_'))
                    for p in (output / 'runs').glob('seed_*/evaluation.json'))
            except Exception as error:
                record['run_export_error'] = str(error)[-2000:]
            try:
                docker('cp', private_id + ':/var/lib/arm/runtime.json', output / 'runtime.json')
                write(output / 'final-inspect.json', json.loads(docker('inspect', *ids.values())))
                docker('logs', '--tail', '1000', private_id, output=output / 'service.log')
            except Exception as error:
                record['diagnostic_export_error'] = str(error)[-2000:]
        try:
            await environment.stop(delete=True)
        except Exception as error:
            record['cleanup_error'] = str(error)[-2000:]
        record['finished_at'] = now()
        write(output / 'summary.json', record)
        print(json.dumps({'event': 'cold_score_finished', 'split': split,
                          'status': record['status'], 'elapsed_seconds': record.get('elapsed_seconds'),
                          'error': record.get('error')}), flush=True)
    return record

async def main():
    binding = {'task_version': '1.2.0', 'job_id': JOB, 'candidate': 'expert_evidence/best_method/select.py',
               'selector_sha256': digest(SOURCE), 'source_files': {}, 'preserved_models': {}}
    for name in ['task.toml', 'environment/Dockerfile', 'environment/Dev.Dockerfile',
                 'environment/docker-compose.yaml', 'environment/arm_static_offline.py',
                 'environment/dev_runtime/service.py', 'environment/dev_runtime/worker.py',
                 'tests/Dockerfile', 'tests/docker-compose.yaml', 'tests/test.sh',
                 'tests/private/runtime/service.py', 'tests/private/runtime/worker.py', 'tests/hidden_assets/manifest.json']:
        binding['source_files'][name] = digest(TASK / name)
    for arm in ['baseline', 'reference']:
        for seed in SEEDS:
            name = f'optimization_evidence/{arm}_runs/runs/seed_{seed}/model/checkpoint'
            binding['preserved_models'][name] = digest(DELIVERY / name)
    write(OUT / 'task-binding.json', binding)
    records = []
    for split in ['dev', 'hidden']:
        records.append(await phase(split))
        write(OUT / 'summary.json', {'status': 'running', 'phases': records, 'job_id': JOB})
        if records[-1]['status'] != 'passed':
            break
    assert all(digest(DELIVERY / p) == value for p, value in binding['preserved_models'].items())
    assert all(digest(TASK / p) == value for p, value in binding['source_files'].items())
    passed = len(records) == 2 and all(r['status'] == 'passed' for r in records)
    write(OUT / 'summary.json', {'status': 'passed' if passed else 'failed', 'phases': records,
          'task_version': '1.2.0', 'job_id': JOB, 'old_models_unchanged': True,
          'new_training_runs': sum(r['reward']['new_training_runs'] for r in records) if passed else None,
          'completed_new_training_seeds': sum(len(r.get('completed_new_training_seeds', [])) for r in records),
          'model_api_requests': 0, 'stability_12h_run': False, 'formal_platform_acceptance': False})
    return 0 if passed else 2

if __name__ == '__main__':
    try:
        setup()
        logging.basicConfig(filename=OUT / 'operator.log', level=logging.INFO)
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        write(OUT / 'setup-failure.json', {'status': 'failed', 'error_type': type(error).__name__,
              'error': str(error)[-3500:], 'job_id': JOB, 'finished_at': now()})
        raise
