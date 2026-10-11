# Current v1.2.0 stability attempt

Job 121852 started the current public Dev pair with the final selector and copied the validated seed models. It stopped after 64.55 seconds because the first monitor implementation required the shared outbox to be empty after score probes; the protocol retains completed responses there. The recorded checks showed running containers, no OOM, no restart, bounded file descriptors, free disk, no active swap, and no GPU. This is a failed monitoring attempt, not a 12-hour stability pass.
