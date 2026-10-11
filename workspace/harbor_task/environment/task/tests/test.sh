#!/bin/sh
set -eu
if [ -d /harbor_task ]; then
    cd /harbor_task
else
    cd "$(dirname "$0")/.."
fi
if [ -d /logs/verifier ]; then
    rm -f /logs/verifier/reward.json /logs/verifier/public-score.json /logs/verifier/failure.json
fi
python3 tests/acquisition_test_contract.py
python3 tests/grader.py --baseline --output solution/artifacts/baseline-check
python3 tests/grader.py --output solution/artifacts/verified --verifier
