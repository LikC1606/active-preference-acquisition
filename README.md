# Active Preference Acquisition

HarborTask for designing preference-pair acquisition methods using the original
Active Reward Modeling quickstart data. The editable function is
`solution/select.py:select_pairs`. Baseline uses random selection and scores 0;
Reference implements the paper ARM-FI D-optimal rule.

## Contents

- `workspace/`: task code, separate Dev/Hidden Docker build contexts and Reference.
- `expert_evidence/`: expert notes, independent GPT/Seed trajectories and final method.
- `optimization_evidence/`: five paired Baseline/Reference training runs, models,
  statistics, cold-scoring and twelve-hour stability evidence.

The large arrays, offline wheels and historical pool archives are stored as
lossless Release assets, rather than Git blobs. Repository files alone do not
include those assets. Download **all assets from the submission release** into
one directory; put the 42 `.gz` volumes in `assets/`, alongside
`task-and-evidence.zip`, `distribution.json`, `restore.py` and `validation.json`.
Then restore a complete independent delivery:

```sh
python3 -B restore.py --output ../delivery-restored
```

Restoration requires only Python's standard library. It verifies every volume
and all 4,008 restored files against the original SHA-256 identities and file
permissions. It performs no training and makes no network requests. See
[DISTRIBUTION.md](DISTRIBUTION.md) for the distribution details and
[the local package guide](optimization_evidence/runtime/LOCAL_PACKAGE.md)
for Docker commands.

## Validation

The local QA report passed 21 checks (including the separately requested QA15
resource supplement), three research content gates and six Harbor checks.
Five-seed cold scores for the final method passed in Dev and Hidden, each within
7,200 seconds. The public Dev container pair passed 43,206.30 seconds of continuous
stability monitoring with 681 checks per method. This monitor does not cover
Hidden; its independent build and cold-score receipts are supplied separately.

Formal platform acceptance remains pending. Raw logs, model identities,
historical failures and the original data/protocol limitations are preserved.

[QA report](optimization_evidence/runtime/qa-review/report.txt) ·
[Expert notes](expert_evidence/专家作业说明文档.md) ·
[Evidence index](optimization_evidence/README.md)
