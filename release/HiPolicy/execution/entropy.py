"""Entropy-guided selection from parallel hierarchical action candidates.

No neural parameters, interpolation, rescaling, or candidate averaging is added.
"""
import math

LOW_THRESHOLD = -6.0
HIGH_THRESHOLD = -5.5
CANDIDATES = 100


def scale_for_entropy(entropy):
    if not math.isfinite(entropy):
        raise ValueError("Non-finite action entropy")
    if entropy < LOW_THRESHOLD:
        return "short", 16
    if entropy < HIGH_THRESHOLD:
        return "mid", 8
    return "long", 0


def expand_batch(tensor, num_samples):
    """Original V2 ordering: [B, N, ...] -> [B*N, ...]."""
    if tensor is None or num_samples == 1:
        return tensor
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    return tensor.unsqueeze(1).repeat(1, num_samples, *([1] * (tensor.ndim - 1))).reshape(
        tensor.shape[0] * num_samples, *tensor.shape[1:])


def unpack_candidates(tensor, batch_size, num_samples):
    """Return [N,B,T,D] without mixing observations when B > 1."""
    return tensor.reshape(batch_size, num_samples, *tensor.shape[1:]).transpose(0, 1)


def select_candidates(candidates, enabled=False):
    """Select one eight-step chunk per batch element from candidate zero.

    Input: unnormalized actions [N,B,24,14]. Variance uses correction=1,
    matching the original torch.var default. Entropy averages T and D only;
    both arms use the same selected scale. With EG off no entropy is computed.
    """
    import torch
    expected = CANDIDATES if enabled else 1
    if candidates.ndim != 4 or candidates.shape[0] != expected or candidates.shape[2:] != (24, 14):
        raise ValueError(f"Expected [{expected},B,24,14] candidates, got {tuple(candidates.shape)}")
    first = candidates[0]
    if not enabled:
        return first[:, 16:24, :], None
    variance = torch.var(candidates, dim=0)
    entropy = (0.5 * torch.log(2 * math.pi * math.e * variance + 1e-6)).mean(dim=(1, 2))
    if not torch.isfinite(entropy).all():
        raise ValueError("Non-finite action entropy")
    starts = torch.where(entropy < LOW_THRESHOLD, 16, torch.where(entropy < HIGH_THRESHOLD, 8, 0))
    indices = starts[:, None] + torch.arange(8, device=first.device)[None, :]
    actions = first.gather(1, indices[:, :, None].expand(-1, -1, first.shape[-1]))
    return actions, entropy


def predict(policy, observations, enabled=False):
    # One call: N candidates are sampled together in the existing model batch.
    n = CANDIDATES if enabled else 1
    result = policy.predict_action(observations, num_samples=n)
    chunks = result["action_chunks"]
    if n == 1:
        chunks = chunks.unsqueeze(0)
    return select_candidates(chunks, enabled)
