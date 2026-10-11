"""ARM-FI dopt_rule, with precomputed last-layer features and logits."""
import torch


def select_pairs(features, logits, k, seed, round_index):
    torch.set_num_threads(2)
    delta = torch.from_numpy(features.copy())
    probability = torch.sigmoid(torch.from_numpy(logits.copy()))
    variance = probability * (1.0 - probability)
    weights = torch.ones_like(variance, requires_grad=True)
    fisher = torch.zeros(delta.shape[1], delta.shape[1])
    for start in range(0, len(delta), 1024):
        end = min(start + 1024, len(delta))
        fisher += (delta[start:end, None, :] * delta[start:end, :, None]
                   * variance[start:end, None, None]
                   * weights[start:end, None, None]).sum(dim=0)
    fisher += 1e-6 * torch.eye(fisher.shape[0])
    gradients = torch.autograd.grad(torch.logdet(fisher), weights, create_graph=True)[0]
    return torch.topk(gradients, k).indices.tolist()
