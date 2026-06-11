"""Mean-of-judges leaf evaluator.

If the hybrid's residual edge over learned judges is evaluator NOISE
(approximation variance around the AB-style value), averaging k decorrelated
judges cuts the noise ~1/sqrt(k) at k forward passes per leaf — the cheap,
training-free test of the consistency hypothesis.

Width handling: members may be POV (OBS_SIZE) or full-obs (OBS_FULL_SIZE)
nets. The ensemble takes the WIDEST input it contains; MCTSvsFixed reads the
first parameter's in-features to decide which obs to write, so members are
ordered full-obs first when any is present. POV members get the [:OBS_SIZE]
prefix slice.

Use: pass a comma-separated --judge-ckpt list to evaluate.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import fastcatan

OBS = fastcatan.OBS_SIZE
FULL = fastcatan.OBS_FULL_SIZE


class JudgeEnsemble(nn.Module):
    def __init__(self, judges: list[nn.Module]):
        super().__init__()
        widths = [next(j.parameters()).shape[1] for j in judges]
        for w in widths:
            if w not in (OBS, FULL):
                raise ValueError(f"judge width {w} unsupported")
        # full-obs members first so the in-features probe sees the widest
        order = sorted(range(len(judges)), key=lambda i: -widths[i])
        self.members = nn.ModuleList([judges[i] for i in order])
        self.widths = [widths[i] for i in order]
        self.in_width = self.widths[0]

    def forward(self, obs: torch.Tensor):
        vs = []
        logits0 = None
        for j, w in zip(self.members, self.widths):
            _lg, v = j(obs[..., :w])
            if logits0 is None:
                logits0 = _lg          # placeholder; judge logits are unused
            vs.append(v)
        return logits0, torch.stack(vs, dim=0).mean(dim=0)
