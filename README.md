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

The complete task is available in [the submission release](https://github.com/LikC1606/active-preference-acquisition/releases/tag/complete-submission).
It has **two attachments**, each below GitHub's per-asset size limit. Download both
into one directory and combine them:

```sh
cat active-preference-acquisition.tar.gz.001 active-preference-acquisition.tar.gz.002 > active-preference-acquisition.tar.gz
mkdir delivery-restored
tar -xzf active-preference-acquisition.tar.gz -C delivery-restored
```

The resulting archive contains all three directories, including complete data,
offline dependencies, models and evidence. SHA-256 identities and file permissions
were verified against the original delivery. Only large identical data and wheel
files use tar hard links; code and evidence remain independent files. Repository
files alone omit the large assets; use the release for a complete submission.

See [the download guide](DISTRIBUTION.md) for hashes and
[the local package guide](optimization_evidence/runtime/LOCAL_PACKAGE.md) for Docker commands.

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
