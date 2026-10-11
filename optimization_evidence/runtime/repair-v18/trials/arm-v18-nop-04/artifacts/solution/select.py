"""Random preference acquisition Starter."""
import numpy as np


def select_pairs(features, logits, k, seed, round_index):
    rng = np.random.RandomState(seed * 1000 + round_index)
    return rng.choice(len(features), k, replace=False).tolist()
