"""Differentiable papercut craft energies on SDXL latents.

Latents are decoded with the tiny TAESD-XL decoder, which is cheap and
differentiable, so the same energies serve as a training regulariser
(``train_lora.py``) and as a sampling-time guidance signal (``sample.py``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB in [0,1], (B,3,H,W) -> CIELAB (D65), differentiable."""
    rgb = rgb.clamp(0, 1)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).clamp_min(1e-6) ** 2.4, rgb / 12.92)
    xyz = torch.einsum("ij,bjhw->bihw", _M.to(rgb), lin) / _WHITE.to(rgb)[None, :, None, None]
    e = 216 / 24389
    f = torch.where(xyz > e, xyz.clamp_min(1e-6) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


class CraftEnergy(torch.nn.Module):
    """E = w_pal * E_palette + w_flat * E_flat + w_edge * E_edge on decoded x0."""

    def __init__(self, palette_lab, taesd, w_pal=1.0, w_flat=1.0, w_edge=0.0, w_ground=0.0,
                 tau_pal=4.0, sigma_flat=8.0, sigma_edge=12.0, size=512):
        super().__init__()
        pal = torch.as_tensor(palette_lab, dtype=torch.float32)
        pal = torch.cat([pal, torch.tensor([[100.0, 0.0, 0.0]])], 0)  # paper-white ground
        self.register_buffer("pal", pal)
        self.taesd = taesd
        self.w_pal, self.w_flat, self.w_edge, self.w_ground = w_pal, w_flat, w_edge, w_ground
        self.tau_pal, self.sigma_flat, self.sigma_edge = tau_pal, sigma_flat, sigma_edge
        self.size = size

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.taesd.decode(latents.to(self.taesd.dtype)).sample  # [-1,1]
        x = (x.float() + 1) / 2
        if self.size and x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        return x

    def terms(self, rgb: torch.Tensor, line_mask: torch.Tensor | None = None,
              ground_mask: torch.Tensor | None = None) -> dict:
        lab = rgb_to_lab(rgb)
        # palette: soft-min Euclidean Lab distance to the Ku Shulan palette (+ white)
        d = torch.cdist(lab.permute(0, 2, 3, 1).reshape(lab.shape[0], -1, 3), self.pal[None].expand(lab.shape[0], -1, -1))
        e_pal = (-self.tau_pal * torch.logsumexp(-d / self.tau_pal, -1)).clamp_min(0).mean(1)
        # flatness: saturating (Welsch) penalty on Lab gradients -> piecewise-constant colour
        gx = lab[..., :, 1:] - lab[..., :, :-1]
        gy = lab[..., 1:, :] - lab[..., :-1, :]
        g2x = (gx ** 2).sum(1)
        g2y = (gy ** 2).sum(1)
        s2 = self.sigma_flat ** 2
        e_flat = (1 - torch.exp(-g2x / s2)).mean((1, 2)) + (1 - torch.exp(-g2y / s2)).mean((1, 2))
        out = {"pal": e_pal, "flat": e_flat}
        if line_mask is not None:
            # edge support: the cut boundary should pass where the content line art is
            g = torch.sqrt(F.pad(g2x, (0, 1, 0, 0)) + F.pad(g2y, (0, 0, 0, 1)) + 1e-6)
            g = F.max_pool2d(g[:, None], 5, 1, 2)[:, 0]
            m = F.interpolate(line_mask.float(), size=g.shape[-2:], mode="area")[:, 0]
            e_edge = ((m * torch.exp(-(g ** 2) / self.sigma_edge ** 2)).sum((1, 2)) / m.sum((1, 2)).clamp_min(1))
            out["edge"] = e_edge
        if ground_mask is not None:
            # figure-on-ground: outside the drawing's silhouette the sheet is bare paper-white
            g = F.interpolate(ground_mask.float(), size=lab.shape[-2:], mode="area")[:, 0]
            dist = torch.sqrt(((lab - self.pal[-1].view(1, 3, 1, 1)) ** 2).sum(1) + 1e-6)
            out["ground"] = (g * dist).sum((1, 2)) / g.sum((1, 2)).clamp_min(1)
        return out

    def forward(self, latents, line_mask=None, ground_mask=None, return_terms=False):
        rgb = self.decode(latents)
        t = self.terms(rgb, line_mask, ground_mask)
        e = (self.w_pal * t["pal"] / 10.0 + self.w_flat * t["flat"] + self.w_edge * t.get("edge", 0.0)
             + self.w_ground * t.get("ground", 0.0) / 10.0)
        return (e, t, rgb) if return_terms else e


@torch.no_grad()
def palette_project(rgb: torch.Tensor, pal_lab: torch.Tensor, pal_rgb: torch.Tensor, mode_r: int = 2):
    """Fast GPU cut projection: nearest palette colour (Lab) + majority filter.

    rgb (B,3,H,W) in [0,1]; pal_lab (K,3); pal_rgb (K,3) in [0,1]. Returns projected rgb.
    """
    lab = rgb_to_lab(rgb)
    b, _, h, w = lab.shape
    d = torch.cdist(lab.permute(0, 2, 3, 1).reshape(b, -1, 3), pal_lab[None].expand(b, -1, -1))
    idx = d.argmin(-1).view(b, h, w)
    if mode_r:
        onehot = F.one_hot(idx, pal_lab.shape[0]).permute(0, 3, 1, 2).float()
        k = 2 * mode_r + 1
        idx = F.avg_pool2d(onehot, k, 1, mode_r, count_include_pad=False).argmax(1)
    return pal_rgb[idx].permute(0, 3, 1, 2)
