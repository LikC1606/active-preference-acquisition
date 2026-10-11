#!/bin/sh
set -eu
cd /harbor_task
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/public-score.json /logs/verifier/failure.json
python3 /harbor_task/tests/acquisition_test_contract.py
python3 /harbor_task/tests/grader.py --baseline --output solution/artifacts/baseline-check
exec python3 /harbor_task/tests/grader.py --output solution/artifacts/verified --verifier
