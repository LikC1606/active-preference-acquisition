# Current v1.2.0 build and stability snapshot

Job 121929 rebuilt the separate Hidden and public Dev contexts with Docker build cache disabled. Both builds and service startup passed within the declared 1,800-second limit. The exact build receipts and monitor source are included.

The public Dev pair then started a twelve-hour monitor. This directory contains an immutable snapshot taken while that monitor was running; it does not claim twelve-hour completion. The monitor reuses job 121710's completed models, checks Baseline and the final selector, and records FD, OOM, restart, disk, shared-memory and logging limits. The service retains completed IPC responses; the monitor checks their expected bounded count instead of requiring the outbox to be empty. No training or model API calls run during monitoring.
