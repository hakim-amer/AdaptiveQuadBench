"""
Offset-free selective estimator ("gain net").

A Mamba backbone reads the causal feature stream and emits, per step and per channel,
  * a gain K_t in (k_min, k_max) for an innovation recursion on the physics residual r_t
        e0_t = (1 - K_t) e0_{t-1} + K_t r_t
    (a selective-SSM channel with A = -1 and input-dependent step Delta_t = -log(1 - K_t),
    i.e. an adaptive-gain Kalman filter whose gain is inferred in context), and
  * convex mixture weights pi_t over the experts [e0, KF_1..KF_n] (fixed-gain Kalman filters).

Output d_t = sum_i pi_t,i e_i,t. Every expert is an offset-free estimator (a normalised,
causal linear filter of the residual with unit DC gain) and the mixture is convex, so d_t is
offset-free for ANY network output: a constant disturbance is recovered without bias. An
unconstrained correction network has no such guarantee, and its low-frequency error is what
the closed loop cannot reject.
"""

import torch
import torch.nn as nn

from icon_mpc.learned.features import BASE_KF, KF_BANK, KF_START, RES_SLICE
from icon_mpc.models.mamba import MambaStack


def gated_scan(K, r, h0, chunk=8):
    """h_t = (1-K_t) h_{t-1} + K_t r_t for (b, T, c) tensors via chunked log-cumsum."""
    out, h = [], h0
    loga = torch.log1p(-K)
    for s in range(0, K.shape[1], chunk):
        e = min(s + chunk, K.shape[1])
        L = torch.cumsum(loga[:, s:e], 1)
        hs = torch.exp(L) * (h[:, None] + torch.cumsum(torch.exp(-L) * K[:, s:e] * r[:, s:e], 1))
        out.append(hs)
        h = hs[:, -1]
    return torch.cat(out, 1)


class GainNet(nn.Module):
    def __init__(self, d_in, d_out, d_model=64, n_layers=2, d_state=16, k_min=2e-3, k_max=0.5,
                 k_init=0.05, mix=True):
        super().__init__()
        self.DO = d_out // 2
        self.nE = 1 + len(KF_BANK) if mix else 1
        self.backbone = MambaStack(d_in, 6 * (1 + self.nE), d_model, n_layers, d_state)
        self.k_min, self.k_max = k_min, k_max
        p = (k_init - k_min) / (k_max - k_min)
        with torch.no_grad():
            self.backbone.head.weight.mul_(0.1)
            b = self.backbone.head.bias
            b.zero_()
            b[:6] = float(torch.logit(torch.tensor(p)))
            if mix:  # start as the base (robust) KF
                b[6:].view(self.nE, 6)[1 + BASE_KF] = 3.0
        self.register_buffer('mu', torch.zeros(d_in))
        self.register_buffer('sd', torch.ones(d_in))
        self.register_buffer('out_sd', torch.ones(self.DO))
        self.kf_idx = [slice(KF_START + 6 * j, KF_START + 6 * j + 6) for j in range(len(KF_BANK))]

    def set_norm(self, mu, sd, out_sd):
        self.mu.copy_(mu)
        self.sd.copy_(sd)
        self.out_sd.copy_(out_sd)

    def _un(self, x, sl):
        return x[..., sl] * self.sd[sl] + self.mu[sl]

    def _heads(self, o):
        K = self.k_min + (self.k_max - self.k_min) * torch.sigmoid(o[..., :6])
        pi = torch.softmax(o[..., 6:].unflatten(-1, (self.nE, 6)), -2)
        return K, pi

    def _combine(self, e0, x, pi):
        experts = [e0] + ([self._un(x, s) for s in self.kf_idx] if self.nE > 1 else [])
        d = (pi * torch.stack(experts, -2)).sum(-2)
        corr = (d - self._un(x, self.kf_idx[BASE_KF])) / self.out_sd[:6]
        z = torch.zeros(*corr.shape[:-1], 2 * self.DO - 6, device=corr.device, dtype=corr.dtype)
        return torch.cat([corr, z], -1)

    def forward(self, x, return_aux=False):
        K, pi = self._heads(self.backbone(x))
        r = self._un(x, RES_SLICE)
        e0 = gated_scan(K, r, torch.zeros_like(r[:, 0]))
        out = self._combine(e0, x, pi)
        return (out, K, pi) if return_aux else out

    def init_state(self, batch, device=None):
        return (self.backbone.init_state(batch, device), torch.zeros(batch, 6, device=device))

    def step(self, x, state):
        bstate, h = state
        o, bstate = self.backbone.step(x, bstate)
        K, pi = self._heads(o)
        h = (1 - K) * h + K * self._un(x, RES_SLICE)
        self.last_gain, self.last_pi = K, pi
        return self._combine(h, x, pi), (bstate, h)
