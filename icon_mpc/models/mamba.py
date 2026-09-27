"""
Minimal pure-PyTorch selective state-space (Mamba-1 style) block.

Two execution modes with identical semantics:
  * forward(x)            : (B, T, D) -> (B, T, D), chunked parallel scan for training
  * step(x_t, state)      : (B, D) -> (B, D), O(1) recurrent update for 100 Hz control

Selective SSM per channel d and state n (diagonal A < 0):
    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * u_t ,   y_t = <C_t, h_t> + D * u_t
dt_t, B_t, C_t are functions of the input, so the effective filter bandwidth adapts to
the context - the property that fixed-gain observers / Kalman filters lack.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DA_MIN = -5.0  # per-step log-decay clamp (exp(-5) ~ full reset); keeps the chunked scan finite


def selective_scan(u, dt, A, B, C, D, h0=None, chunk=16):
    """u, dt: (b, T, d); A: (d, n); B, C: (b, T, n); D: (d,). Returns y (b, T, d), h_T (b, d, n).

    Within a chunk h_t = exp(L_t) (h0 + sum_{r<=t} exp(-L_r) x_r), L = cumsum(dt*A); with the
    per-step clamp |L| <= 16*5 = 80 < 88 so exp(-L) stays finite in float32.
    """
    assert chunk * -DA_MIN < 88, 'chunk too long for float32 exp(-L)'
    b, T, d = u.shape
    n = A.shape[1]
    h = torch.zeros(b, d, n, device=u.device, dtype=u.dtype) if h0 is None else h0
    ys = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        dA = (dt[:, s:e, :, None] * A).clamp(min=DA_MIN)
        x = dt[:, s:e, :, None] * B[:, s:e, None, :] * u[:, s:e, :, None]
        L = torch.cumsum(dA, dim=1)
        hs = torch.exp(L) * (h[:, None] + torch.cumsum(torch.exp(-L) * x, dim=1))
        ys.append(torch.einsum('btdn,btn->btd', hs, C[:, s:e]) + D * u[:, s:e])
        h = hs[:, -1]
    return torch.cat(ys, dim=1), h


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_min=1e-3, dt_max=1e-1, selective=True):
        super().__init__()
        self.selective = selective  # False -> LTI SSM ablation (dt, B, C independent of the input)
        d_inner = expand * d_model
        self.d_inner, self.d_state, self.d_conv = d_inner, d_state, d_conv
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # softplus^-1
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        if not selective:
            self.B0 = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
            self.C0 = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))

    def _ssm_params(self, x):
        if not self.selective:
            shp = x.shape[:-1]
            return (F.softplus(self.dt_proj.bias).expand(*shp, -1), self.B0.expand(*shp, -1),
                    self.C0.expand(*shp, -1))
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        return F.softplus(self.dt_proj(dt)), B, C

    def forward(self, x, return_dt=False):
        T = x.shape[1]
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, B, C = self._ssm_params(u)
        y, _ = selective_scan(u, dt, -torch.exp(self.A_log), B, C, self.D)
        out = x + self.out_proj(y * F.silu(z))
        return (out, dt) if return_dt else out

    def init_state(self, batch, device=None):
        device = device or self.D.device
        return (torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device),
                torch.zeros(batch, self.d_inner, self.d_state, device=device))

    def step(self, x, state):
        conv_buf, h = state
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        window = torch.cat([conv_buf, u[..., None]], dim=-1)
        u = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        dt, B, C = self._ssm_params(u)
        A = -torch.exp(self.A_log)
        h = torch.exp((dt[..., None] * A).clamp(min=DA_MIN)) * h + dt[..., None] * B[:, None] * u[..., None]
        y = (h * C[:, None]).sum(-1) + self.D * u
        return x + self.out_proj(y * F.silu(z)), (window[..., 1:], h)


class MambaStack(nn.Module):
    def __init__(self, d_in, d_out, d_model=64, n_layers=2, d_state=16, selective=True):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, selective=selective) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))

    def init_state(self, batch, device=None):
        return [l.init_state(batch, device) for l in self.layers]

    def step(self, x, states):
        h = self.inp(x)
        new = []
        for l, s in zip(self.layers, states):
            h, s = l.step(h, s)
            new.append(s)
        return self.head(self.norm(h)), new
