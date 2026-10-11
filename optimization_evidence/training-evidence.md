# Acquisition training evidence

The only optimization surface is `solution/select.py:select_pairs`. Baseline
is random acquisition; Reference implements the public ARM-FI `dopt_rule`.
Both use the same frozen model fitting, data split, annotation budget and five
seeds. Each seed trained from initialization for 40 rounds and 40000 Adam
updates. The Reference selector was checked against upstream for exact selected
indices and order. Every final checkpoint was loaded in the trusted container
and reproduced its metrics.

`experiment_plan.json` is a copy of the protocol frozen before calibration,
not a retrospectively chosen experimental plan. The private protocol hash is
`1d2828746ca304e36f93c7a32935f8b46afbe1715fe09a71117b79b0bd2a9fcf`.
Training jobs were 115560 and 116024; the latter resumed missing computation
without repeating completed rounds. Historical compressed-protocol results
remain in their original directories and are excluded from these anchors.

All five Baseline and all five Reference results are included, with complete
final models, 40-round metrics and update accounting. Baseline mean error is
0.6799700626090104; Reference mean error is 0.4984071729759192. Its score is
0.2670160049935782. Gain is 4.7413 times the Baseline sample standard deviation:
above 3 sigma, below 5 sigma, with five repeats and platform review required.
No additional ablation is necessary for this single acquisition change.

The source is the original roughly 10GB quickstart data. Public assets contain
only train/dev embeddings; reward targets remain private. The task uses
ten-response evaluation, not the paper's independent 500-response test.
Training uses public `train_BT` defaults (1000 full-batch updates, lr 0.001);
these are not claimed to be the paper's undisclosed experiment parameters.
See `alignment.md` and `data_manifest.json` for sources and limitations.

Historical native Harbor job 116184 checked both agent-facing `solution/solve.sh` and the
separate verifier. Both Baseline and Reference produced the scientific scores
with zero new training. The allocated VM had 8 CPUs, 0 GPUs and 40960 MiB RAM.
The agent and verifier see only an evaluation inbox and a read-only outbox;
private rewards, training models, evaluator code and Docker remain outside.

For current deployment, build and start the Compose services from the extracted
three-directory submission root. The public main container and private evaluator
share IPC; task data needs no host mount, external broker or internal service.
Pinned dependencies install offline from bundled wheels. The independent
Reference stays outside the Agent workspace. See `runtime/LOCAL_PACKAGE.md`
for exact build, public scoring, cached-result and packaging commands.

Current v14 upstream/isolation probes, final offline no-cache build, official
Harbor five-seed Baseline/Reference/best fresh acceptance and twelve-hour runtime
stability passed. Scores reproduce the recorded science and all cases finish
within the declared 7,200-second budget. Complete receipts and checkpoints are
in `runtime/acceptance-v14/`; original failed attempts remain unchanged. Both
required model lanes were rerun independently for at least ten effective hours;
every completed proposal-to-score loop took at most 7,200 seconds. Old trajectories
are preserved as superseded evidence and are not counted in the new duration.
Formal Hidden, secure-asset and license review remain the platform's responsibility.

证据现按 `baseline_runs/seed_*` 和 `reference_runs/seed_*` 保存。每个 `result.json` 为原记录索引；原配置、计数和指标位于 `records/`。`model/reload.log` 是原始 evaluation 收据的逐字节副本，不是新执行日志。历史路径对照见 `evidence_paths.json`；完整目录说明见 `README.md`。
