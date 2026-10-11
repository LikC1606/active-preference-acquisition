# Active Preference Acquisition

A CPU-only Harbor research task for improving how a reward model chooses preference pairs to label. Implement one selection function; the evaluator keeps the data, labeling budget, model and training procedure fixed.

The task uses the original Active Reward Modeling quickstart data. The random Starter is the **Baseline**, with score **0**. The paper's **ARM-FI D-optimal method** is the **Reference**. Reference demonstrates an improvement under the task protocol; its score is not a ceiling or a passing threshold.

## What you optimize

Edit [`workspace/harbor_task/solution/select.py`](workspace/harbor_task/solution/select.py):

```python
def select_pairs(features, logits, k, seed, round_index):
    # Return k distinct candidate indices as a list of Python integers.
    ...
```

Each call receives 22,500 candidate pairs, their 64-dimensional feature differences and predicted reward differences. Select 500 pairs to label. Calls run in fresh isolated processes, so selection must use the supplied inputs without carrying state between rounds. Candidate rows are shuffled; consecutive rows do not identify prompt groups.

Each seed runs 40 acquisition rounds with 20,000 labels and 40,000 Adam updates. Round 0 is a fixed random bootstrap; your function controls rounds 1–39. Evaluation uses five seeds: `101`, `202`, `303`, `404` and `505`.

See the [task instructions](workspace/harbor_task/instruction.md) for the full interface, allowed imports and submission rules.

## Scoring

The evaluator measures response-ranking error as one minus the mean Spearman correlation with oracle rankings. Lower error is better. The final score compares the five-seed mean error with the random Baseline:

```text
score = (B - mean_error) / B
Public Dev B = 0.6799700626090104
```

A positive score improves on random selection; a negative score is worse. Scores are continuous and are not clipped. Hidden evaluation uses the same formula with a Baseline measured on the separate Hidden split.

| Method | Selection rule | Recorded Dev score |
| --- | --- | ---: |
| Baseline | Unmodified random Starter | 0.000000 |
| Reference | Paper ARM-FI D-optimal rule | 0.267016 |
| Best recorded agent selector | Fisher-weighted covariance leverage | 0.249275 |

These are task results, not the paper's reported benchmark numbers. The final agent selector also scored `0.244336` in a local Hidden cold run. [Per-seed results and comparison](optimization_evidence/comparison_summary.json) are included.

## Download the complete task

**Use the [complete release](https://github.com/LikC1606/active-preference-acquisition/releases/tag/complete-submission) to run the task.** Git contains code, documentation and evidence indexes; large data, dependency wheels and model checkpoints are supplied in the release.

Download both attachments into the same directory:

- [active-preference-acquisition.tar.gz.001](https://github.com/LikC1606/active-preference-acquisition/releases/download/complete-submission/active-preference-acquisition.tar.gz.001)
- [active-preference-acquisition.tar.gz.002](https://github.com/LikC1606/active-preference-acquisition/releases/download/complete-submission/active-preference-acquisition.tar.gz.002)

The complete archive is about **3.69 GB**. It is split into two parts to fit GitHub's 2 GiB per-attachment limit. Combine, verify and extract:

```sh
cat active-preference-acquisition.tar.gz.001 active-preference-acquisition.tar.gz.002 > active-preference-acquisition.tar.gz
sha256sum active-preference-acquisition.tar.gz
mkdir delivery-restored
tar -xzf active-preference-acquisition.tar.gz -C delivery-restored
cd delivery-restored
```

The SHA-256 must be:

```text
db2f9d65ab8722eb29f11a9f29a549e947157399e893b6a00914aa43c41c7531
```

The archive restores all 4,008 files, including data, offline dependencies, models and evidence. File hashes and permissions were verified. See [DISTRIBUTION.md](DISTRIBUTION.md) for individual attachment hashes.

## Run locally

You need **Docker with Compose v2**, **8 CPUs**, **36 GiB of container memory** and **no GPU**. A complete five-seed candidate evaluation has a two-hour budget; run one evaluation at a time.

Dependencies install from bundled wheels. For an offline build, first ensure the public base image `python:3.11.11-slim-bookworm` is available locally. Run these commands from the extracted root:

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml build --no-cache

docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml up -d --no-build
```

Edit `workspace/harbor_task/solution/select.py`, then evaluate it:

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml \
  exec -T main sh -c 'cd /harbor_task && ./solution/solve.sh --output solution/artifacts/current'
```

Inside the public container, results are written to `solution/artifacts/current/reward.json` and `observations.json`. Failed attempts write `failure.json` with the cause and repair instructions. A timeout is a failed attempt, not a zero score.

For contract checks, recorded-result replay, container cleanup and Harbor integration, see the [local package guide](optimization_evidence/runtime/LOCAL_PACKAGE.md). Local Harbor 0.22.0 validation uses the documented offline-provider adapter.

## Package layout

```text
workspace/
  harbor_task/          Task instructions, Starter, environments and tests
  reference/            Paper ARM-FI implementation
expert_evidence/        Expert notes, independent GPT/Seed trajectories and final selector
optimization_evidence/  Per-seed runs, checkpoints, comparisons and runtime evidence
```

Only `workspace/harbor_task/` is the Agent workspace. Keep Reference, expert evidence and private runtime assets outside it. The public Dev environment and separate Hidden verifier have independent build contexts; the verifier receives the submitted selector source and computes its score independently.

## Validation and scope

- Local QA passed 21 checks, including the separately requested resource supplement; three content gates and six Harbor checks also passed.
- The final selector completed fresh five-seed Dev and Hidden scoring in **5,196.04 s** and **4,769.36 s**, respectively. Both fit the 7,200-second scoring budget.
- The public Dev container pair passed **12 hours** of continuous monitoring: 681 Baseline probes and 681 final-selector probes, with no errors, OOM events or container restarts. Monitoring reused validated models and performed no new training. It does not cover Hidden.

The task uses precomputed embeddings from the roughly 10 GB original quickstart data and ten responses per prompt. It does not reproduce the paper's independent 500-response test or claim a full reproduction of the paper's experiments. Protocol details and limitations are recorded in the [experiment plan](optimization_evidence/experiment_plan.json).

**Formal platform acceptance remains pending.** Local checks and local Hidden runs are recorded evidence, not platform approval.

[QA report](optimization_evidence/runtime/qa-review/report.txt) · [Expert notes](expert_evidence/专家作业说明文档.md) · [Evidence index](optimization_evidence/README.md)
