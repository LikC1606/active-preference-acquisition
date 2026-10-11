#!/bin/sh
set -eu
task_root="${ARM_TASK_ROOT:-/harbor_task}"
cp "$(dirname "$0")/select.py" "$task_root/solution/select.py"
cd "$task_root"
exec ./solution/solve.sh "$@"
