"""Native Harbor provider with a bounded operator acquisition channel."""
from pathlib import Path

from arm_native_static_network import StaticOfflineDocker


class AcquisitionDocker(StaticOfflineDocker):
    def __init__(self, *args, broker_root, **kwargs):
        root = Path(broker_root).resolve()
        mounts = list(kwargs.pop("mounts", None) or [])
        mounts.extend([
            {"type": "bind", "source": str(root / "inbox"), "target": "/opt/arm-evaluator/inbox"},
            {"type": "bind", "source": str(root / "outbox"), "target": "/opt/arm-evaluator/outbox",
             "read_only": True},
        ])
        super().__init__(*args, mounts=mounts, **kwargs)
