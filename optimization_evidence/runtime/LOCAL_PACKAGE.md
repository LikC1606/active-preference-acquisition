# Local Package: Active Preference Acquisition

Edit `solution/select.py:select_pairs` to choose the preference pairs used for
reward-model training. Each call selects 500 indices from 22,500 pairs. Data,
labeling, model fitting and evaluation remain fixed. The [task
instructions](../../workspace/harbor_task/instruction.md) define the interface.

| Method | Selection rule | Recorded score |
| --- | --- | ---: |
| Baseline | Untouched random selector | 0.000000 |
| Reference | Paper ARM-FI D-optimal rule | 0.267016 |
| Best local method | GPT Fisher-weighted covariance leverage | 0.249275 |

The score is `(B - mean_error) / B`, with `B = 0.6799700626090104`. These values
come from five seeds: 101, 202, 303, 404 and 505. Each seed uses 40 rounds,
20,000 labels and 40,000 Adam updates. The task uses the original roughly 10 GB
quickstart data and ten-response dev evaluation; it does not reproduce the
paper's independent 500-response test.

## Build and run

Extract all three top-level directories together: `workspace/`,
`expert_evidence/` and `optimization_evidence/`. Run these commands from that
extracted root on an operator host with Docker and Compose v2:

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml build --no-cache

docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml up -d --no-build

docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml \
  exec -T main sh -c 'cd /harbor_task && ./solution/solve.sh --output solution/artifacts/current'
```

The public `main` container contains the Starter, embeddings and public scoring
commands. The private `evaluator` container contains the reward targets and
training runtime. It builds from `environment/Dev.Dockerfile` with `environment/`
as its entire context and contains train/Dev targets only. They share an IPC
namespace; neither uses a task-data volume,
a network service, an internal registry or an internal account. The public
Dockerfile installs pinned dependencies from bundled wheels without network
access. The public base image `python:3.11.11-slim-bookworm` must already be
present for an offline build, or pulled from the public registry beforehand.

The public result files are `/harbor_task/solution/artifacts/current/reward.json`
and `observations.json`. For the contract tests, protected Baseline calibration
and Candidate evaluation:

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml \
  exec -T main sh -c 'cd /harbor_task && ./tests/test.sh'
```

The task directory contains all build inputs. Harbor uses
`environment/docker-compose.yaml` for the Agent's public Dev
workflow. The final verifier builds independently from `tests/Dockerfile`, with
`tests/` as its entire context. It receives `/harbor_task/solution` through the
declared artifact transfer and runs `/tests/test.sh`. Hidden embeddings and
rewards are copied only into that verifier image. Its 4,188 prompts are disjoint
from training and Dev; its Baseline is calibrated on Hidden rather than using
the Dev denominator above. Hidden scores are not available through the public
Dev channel.

For local Harbor 0.22.0 validation, the default Docker provider rejects
`no-network`. The bundled `environment/arm_static_offline.py` adapter keeps
all Compose services on `network_mode: none`. Its use is explicit in the
recorded Trial configuration; this is not a test of the default provider.

The GPT selector groups shuffled candidates into consecutive blocks of 45.
Those blocks are not prompt groups, so its result does not demonstrate one
selected pair per prompt. Original model responses remain in the attempt
records; the official trajectory includes this correction.

Keep Reference, expert evidence and private runtime assets outside the Agent
workspace. To release the local containers after evaluation:

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml down
```

## Check completed results

This command uses only the Python standard library. It checks the recorded
five-seed results without starting Docker or training:

```sh
python3 -B optimization_evidence/runtime/private/evaluate.py \
  --selector expert_evidence/best_method/select.py \
  --output ../best-method-replay --cached-only
```

Use `workspace/harbor_task/solution/select.py` for Baseline or
`workspace/reference/select.py` for Reference. The check verifies source and
protocol hashes, checkpoint identities, finite metrics, completed label/update
budgets and successful reload receipts. Its score is computed from recorded
metrics. It is not a fresh scoring-budget test.

Keep replay output outside the submission root. Cache files and locks default
to `../acquisition-runtime/`. Private output files are created with mode `0600`;
existing permissions are preserved. `private/evaluate.sh` preserves the working
directory and accepts `TASK_PYTHON` for its Python executable.

## Check and package

Use Python 3.12 with Harbor installed. After intentional edits, refresh the
manifest, then check the extracted package:

```sh
python3 -B optimization_evidence/runtime/operator/prepare_arm_acquisition_delivery.py \
  --manifest-only --delivery .

python3 -B optimization_evidence/runtime/operator/package_arm_acquisition.py \
  --check-only
```

The manifest records file identities; refreshing it does not validate experiment
claims. Packaging checks those identities, Harbor's file collection, shell
syntax, the private Hidden manifest, executable permissions and ZIP CRC. To create a
new archive outside the submission root:

```sh
python3 -B optimization_evidence/runtime/operator/package_arm_acquisition.py \
  --require-author-acceptance --output ../submission-repacked.zip
```

Existing ZIPs and adjacent reports are not overwritten. The default is level-1
DEFLATE. `--compression-level 6` trades CPU time for size; `--compression stored`
disables compression. `--delivery /path/to/extracted-root` selects another root.

## Resource and failure contract

One complete Candidate evaluation has 7,200 seconds, including lock waiting
and all five seeds. Each deployment permits one evaluation at a time and uses
8 CPUs, 36 GiB and 0 GPUs: trainer 6 CPUs/32 GiB, selector 2 CPUs/4 GiB.
Each selector call has a 60-second limit. Build has 1,800 seconds; Verifier has
7,500 seconds for contract tests, cached Baseline and Candidate evaluation.
Setup, Job and Agent allowances are separate runner settings.

Every attempt clears old result files. Observations are published atomically,
then the score is committed last. Failures write `failure.json` with a category
and repair action. A non-cheating hard gate also returns invalid, non-rankable
`-1`; format, timeout, resource and infrastructure failures have no normal score.

## Current v1.2.0 validation

The current final selector passed a fresh five-seed Dev and Hidden cold score in
job `121710`: 5,196.04 seconds and score `0.24927529058448603` on Dev;
4,769.36 seconds and score `0.24433649204862387` on Hidden. Each phase trained
all five seeds from an empty candidate state and stayed within the 7,200-second
scoring budget. The complete receipts are in
`cold-score-v20/job-121710/`.

Job `121929` rebuilt both current Docker contexts with `--no-cache`, then passed
43,206.30 seconds of continuous monitoring of the public Dev container pair.
All 681 Baseline probes and 681 best-method probes passed, with no errors, OOM
or container restarts. Final checks confirmed that task files and preserved
models were unchanged. The monitor reused the validated cold-score models,
performed zero new training and made zero model requests. Completed receipts
are in `stability-v21/job-121929-final/`; the earlier running snapshot and failed
job `121852` remain separately preserved. This monitoring does not cover Hidden.

## Historical v14 acceptance

Author acceptance for the earlier Dev deployment passed on the five-seed
scientific protocol:

| Method | Complete verifier time (minutes) | New training runs |
| --- | ---: | ---: |
| baseline | 88.274 | 5 |
| reference | 86.649 | 5 |
| best | 88.222 | 5 |

Each case started with no candidate cache, used the official Harbor provider,
artifact handler and separate verifier, reloaded all final checkpoints and
reproduced its frozen score within 7,200 seconds. The final instruction snapshot
also passed an offline no-cache build.

The monitored container pair survived twelve actual hours. All process FD
counts, OOM/health, disk space, shared memory, log rotation, resource limits and
fixed container/image identities passed. `stability-runtime-bindings.json`
checks the installed runtime and data against the package. The monitored instruction differs only in prose; its interface, code blocks and hard boundaries match.
The recorded image IDs identify the exact pair tested; this
does not claim that every later build has the same image ID.

Receipts, checkpoints and failed/superseded attempts are in `acceptance-v14/`.
Those records do not validate the version 1.2.0 Hidden verifier. Its build,
calibration and Trial evidence is stored separately in `repair-v18/`.
Formal Hidden, security and license review remain the platform's responsibility.
