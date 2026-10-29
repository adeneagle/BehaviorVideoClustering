"""Temporal smoothing of per-frame part probabilities.

The network labels each frame on its own, so paw labels flicker between
frames. temporal_smooth() averages the softmax over neighbouring frames
(centred window, edges renormalised), which is what run_video.py would apply
along a tracked crop sequence.
"""

import torch


def kernel(sigma=None, radius=None, device="cpu"):
    """1-D weights over offsets -radius..radius: Gaussian if sigma else uniform."""
    radius = radius if radius is not None else int(round(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    w = torch.exp(-0.5 * (t / sigma) ** 2) if sigma else torch.ones_like(t)
    return w / w.sum()


def temporal_smooth(prob, sigma=None, radius=None):
    """prob [T,C,H,W] (softmax) -> same shape, averaged along T.

    Frames near the ends use the part of the window that exists.
    """
    w = kernel(sigma, radius, prob.device)
    r = (len(w) - 1) // 2
    T = prob.shape[0]
    flat = prob.reshape(T, -1)
    out = torch.zeros_like(flat)
    norm = torch.zeros(T, 1, device=prob.device)
    for o in range(-r, r + 1):
        lo, hi = max(0, -o), min(T, T - o)
        out[lo:hi] += w[o + r] * flat[lo + o : hi + o]
        norm[lo:hi] += w[o + r]
    return (out / norm).reshape(prob.shape)
