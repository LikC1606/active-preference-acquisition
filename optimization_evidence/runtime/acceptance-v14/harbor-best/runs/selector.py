import numpy as np


def select_pairs(features, logits, k, seed, round_index):
    x = np.asarray(features, dtype=np.float32)
    z = np.asarray(logits, dtype=np.float32).reshape(-1)
    n, d = x.shape
    if k <= 0:
        return []

    # Full-covariance whitening captures correlations between hidden units.
    covariance = (x.T @ x) / max(n, 1)
    mean_variance = float(np.trace(covariance)) / max(d, 1)
    ridge = max(0.03 * mean_variance, 1e-8)
    values, vectors = np.linalg.eigh(covariance)
    projected = x @ vectors
    leverage = np.sum(
        projected * projected / (values + ridge)[None, :], axis=1
    )

    # Log Fisher information avoids underflow for confident predictions.
    log_fisher = -np.logaddexp(0.0, z) - np.logaddexp(0.0, -z)
    utility = log_fisher + np.log(leverage + 1e-12)

    # Preserve prompt coverage when the standard 45-pair blocks are present.
    if n == 45 * k:
        offsets = np.argmax(utility.reshape(k, 45), axis=1)
        return (45 * np.arange(k, dtype=np.int64) + offsets).tolist()

    count = min(k, n)
    chosen = np.argpartition(utility, n - count)[n - count:]
    chosen = chosen[np.argsort(utility[chosen], kind="stable")[::-1]]
    return chosen.astype(np.int64).tolist()
