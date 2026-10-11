#!/bin/sh
set -eu
arm_evaluator_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${TASK_PYTHON:-python3}" -B "$arm_evaluator_dir/evaluate.py" "$@"
