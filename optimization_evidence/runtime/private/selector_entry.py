"""Read a public pool and invoke the candidate in its isolated container."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("candidate", "/selector.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with np.load("/pool.npz", allow_pickle=False) as pool:
    features, logits = pool["features"], pool["logits"]
    features.flags.writeable = logits.flags.writeable = False
    result = module.select_pairs(features, logits, int(pool["k"]),
                                 int(pool["seed"]), int(pool["round_index"]))
    if isinstance(result, np.ndarray):
        result = result.tolist()
Path("/output/indices.json").write_text(json.dumps(result, allow_nan=False))
