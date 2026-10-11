# Active Preference Acquisition

## Goal

Implement `select_pairs` in `solution/select.py` to choose preference pairs for
reward-model training. Improve the final model's response ranking over the
random baseline by changing pair selection. Data, labeling budgets, model
architecture and training are fixed. A valid submission completes all five
seeds with a finite score and passes the interface and resource checks.

## Task Setting

The container starts in `/harbor_task` with NumPy, SciPy and PyTorch installed.
Evaluation is CPU-only and needs no network. Use these fixed paths:

- `/harbor_task/solution/select.py`: your selector, initialized from the
  random Starter Code.
- `/harbor_task/environment/starter/select.py`: the read-only Starter Code.
- `/harbor_task/environment/public_assets/`: frozen 2048-dimensional embeddings
  for 33,500 training prompts and 4,188 separate dev prompts, with ten responses
  per prompt.
- `/harbor_task/environment/baseline.json`: the fixed Baseline error.
- `/harbor_task/solution/artifacts/`: writable evaluation results.

The untouched random Starter Code has a five-seed public dev error of
`0.6799700626090104` and a normalized score of `0`.

The public dev split is available for candidate evaluation. Preference labels,
oracle rewards and private evaluation assets remain outside the workspace.
The platform runs the formal Hidden evaluation separately.

Each seed has 40 acquisition rounds and a budget of 20,000 labels. Each round
samples 500 prompts, forms 22,500 response pairs and labels 500 pairs. Round
`0` uses a fixed random bootstrap; your selector chooses the pairs in rounds
`1`–`39`. The oracle samples labels with probability
`sigmoid(reward_a - reward_b)`, using the raw reward difference.

After each round, the evaluator trains a fresh `2048-64-64-64-1` GELU MLP on all
labels collected so far. The output head is linear, without bias. Each round
uses 1,000 full-batch Adam updates at learning rate `0.001`, for 40,000 updates
per seed. The final-round model is scored.

## Objective and Metrics

The primary metric, `one_minus_spearman`, measures response-ranking error. For
each dev prompt, the evaluator computes the Spearman correlation between model
and oracle rankings of its ten responses. Error is one minus the mean
correlation across all 4,188 dev prompts. Ties use average ranks; constant ranks
have zero correlation. Lower error is better.

The evaluator scores the final model for seeds `101`, `202`, `303`, `404` and
`505`. With `mean_error` equal to the mean of these five errors, the reward is:

```text
score = (B - mean_error) / B
B = 0.6799700626090104
```

`B` is the five-seed dev error of the untouched random Starter Code, recorded
in `environment/baseline.json`. Final Hidden evaluation uses the same metric
and formula, with `B` measured for the untouched Starter on the private Hidden
split. The verifier evaluates the final model on that split only after
submission; public commands return Dev scores.

The Starter's normalized score is `0` on either split. Maximize `score`:
positive values improve on Baseline, negative values are worse, and error `0`
gives score `1`. The score is continuous and is not clipped. `best_of_10` is
reported as a diagnostic and does not affect the reward.

## Allowed Scope

Only `solution/select.py` is scored. Keep all selection logic in this file and
define:

```python
def select_pairs(features, logits, k, seed, round_index):
    ...
```

The evaluator calls it with these arguments:

| Argument | Value |
| --- | --- |
| `features` | Read-only float32 array, shape `(22500, 64)`. Each row is a pair's difference in last-hidden-layer features. |
| `logits` | Read-only float32 array, shape `(22500,)`. Predicted reward differences for the same pairs, in the same order. |
| `k` | Number of pairs to label: `500`. |
| `seed` | One of `101`, `202`, `303`, `404` or `505`. |
| `round_index` | Acquisition round, from `1` through `39`. |

Return exactly `k` distinct Python integers in a list. Each index selects the
same row in `features` and `logits` and must satisfy
`0 <= index < len(features)`. The trainer adds pairs in the returned order.
Candidate rows are shuffled; consecutive rows do not identify prompt groups.

Each call runs in a fresh isolated process, with no state carried between
rounds. Base each selection on the supplied inputs. During development, you
may read public assets and write results under `solution/artifacts/`.

## Hard Boundaries

1. Keep `solution/select.py` at most 64 KiB and satisfy the interface above.
2. Do not edit other task files, including `solution/submission.json`. Data,
   candidate pools, oracle labeling, model architecture, training, seeds and
   evaluation are fixed.
3. Do not access labels for unselected pairs, private reward targets, Hidden
   assets or private reference implementations. Do not edit the grader or
   fabricate scores.
4. Run at most one evaluation job at a time. A complete evaluation must finish
   within 7,200 seconds (2 hours), using at most 8 CPUs, 0 GPUs and no network
   access.
5. The trainer uses 6 CPUs and 32 GiB. Each selector call has 2 CPUs, 4 GiB
   and 60 seconds.
6. Allowed imports are `numpy`, `scipy`, `torch`, `math`, `random`,
   `statistics`, `itertools`, `functools`, `collections`, `heapq`, `operator`,
   `typing`, `dataclasses`, `copy`, `enum`, `warnings` and `time`. Dynamic
   imports and external file access from the selector are forbidden. The
   evaluator checks the source and executes it in isolation.
7. A non-cheating constraint violation returns an invalid, non-rankable `-1`.
   Format errors, timeouts, resource limits and infrastructure failures produce
   `failure.json` with repair instructions, without a normal score.
8. Submit your own selector through the provided `solution/submission.json`.
   Results must match the selector's source hash and the fixed protocol.
   Self-reported scores and supplied checkpoints are not accepted.

## Submission Instructions

The scaffold includes `solution/submission.json`:

```json
{
  "schema": "arm.acquisition.submission.v1",
  "selector_path": "solution/select.py"
}
```

Leave it unchanged. `selector_path` is relative to `/harbor_task` and points to
the source the evaluator loads.

Run a five-seed public dev evaluation from `/harbor_task`:

```sh
./solution/solve.sh --output solution/artifacts/current
```

The command evaluates the current selector and writes:

| File | Contents |
| --- | --- |
| `solution/artifacts/current/reward.json` | Normalized score. |
| `solution/artifacts/current/observations.json` | Individual seed errors. |

Result files are cleared before each attempt. A successful run publishes
observations first and commits the score last. If the attempt fails, read
`failure.json` for the cause and repair instructions.

To run the contract tests, cached Baseline check and candidate evaluation:

```sh
./tests/test.sh
```

If either command reports that the scoring service is unavailable, retry after
the service has been restored.

## Workflow & Iteration

This is an anytime task. Evaluate a working selector early, then continue
research within the configured outer deadline. The final verifier scores the
source left in `solution/select.py`.

For each trial:

1. Save the current best selector and edit `solution/select.py`.
2. Run `./solution/solve.sh --output solution/artifacts/current`.
3. Compare `reward.json` and `observations.json` with the current best run.
4. Keep the candidate if it improves on your best; otherwise restore the
   previous selector.

Each evaluation has a 2-hour scoring budget. Matching source and protocol
hashes allow completed runs to be reused. Interrupted training resumes from
the last committed round. A timeout is a failed attempt, not a zero score.

Before the outer deadline, stop new experiments and leave the best evaluated
selector in `solution/select.py`.

## Completion Criteria

You are done only when:

- `solution/submission.json` has the required contents and points to your selector.
- All five seeds have finished and produced a finite score.
- The selector passes the interface, source size, time and resource checks.
- No protected files have been changed and no private assets have been accessed.
- `solution/select.py` contains the best evaluated version you intend to submit.

The final verifier freezes the submitted source and reloads the final
checkpoints in a separate container. It computes the score independently.
