import numpy as np

def select_pairs(features, logits, k, seed, round_index):
    rng = np.random.RandomState(seed * 1000 + round_index)
    n = len(logits)

    z = logits.astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-z))
    grad_mag = sig * (1.0 - sig)

    # BADGE gradient embedding scaled by cross-entropy uncertainty
    # Uncertainty is higher when logit is near 0 (ambiguous pairs)
    grad_emb = features.astype(np.float64) * grad_mag[:, np.newaxis]

    # Pool size scales with round: smaller early (focus on uncertainty),
    # larger later (more diversity needed as model improves)
    pool_factor = min(10 + round_index // 3, 25)
    pool_size = min(pool_factor * k, n)

    grad_norms = np.linalg.norm(grad_emb, axis=1)
    top_pool_idx = np.argpartition(grad_norms, -pool_size)[-pool_size:]
    pool_emb = grad_emb[top_pool_idx]
    pool_norms = grad_norms[top_pool_idx]
    m = len(top_pool_idx)

    # k-means++ selection with hybrid: 90% farthest-point (greedy),
    # 10% stochastic sampling proportional to min distance squared
    # This balances exploration (BADGE's randomized k-means++) with
    # exploitation (greedy maximum-diversity picking)
    selected = []

    # First pick: max gradient norm (most uncertain)
    first = np.argmax(pool_norms ** 2)
    selected.append(first)

    min_sq_dists = np.full(m, np.inf)
    diff = pool_emb - pool_emb[first:first + 1]
    sq_dists = np.sum(diff * diff, axis=1)
    min_sq_dists = np.minimum(min_sq_dists, sq_dists)

    for i in range(1, k):
        scores = min_sq_dists.copy()
        scores[selected] = -np.inf

        # Mix greedy (argmax) with stochastic k-means++ sampling
        if rng.random() < 0.9:
            best = np.argmax(scores)
        else:
            # Sample proportional to squared distance (true k-means++)
            pos_scores = np.maximum(scores, 0.0)
            total = pos_scores.sum()
            if total > 0:
                probs = pos_scores / total
                best = rng.choice(m, p=probs)
            else:
                best = np.argmax(scores)

        selected.append(best)

        diff = pool_emb - pool_emb[best:best + 1]
        sq_dists = np.sum(diff * diff, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)

    result = top_pool_idx[selected].tolist()
    return result
