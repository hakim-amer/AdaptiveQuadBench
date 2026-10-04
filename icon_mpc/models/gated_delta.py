"""
Minimal pure-PyTorch Gated DeltaNet block (Yang et al., gated delta rule) with the same
interface as MambaStack: forward(x) for training, step(x_t, state) for 100 Hz control.

Per head, a matrix state S in R^{dv x dk} is updated by the gated delta rule
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ,   o_t = S_t q_t
with ||k_t|| = 1, alpha_t in (0, 1), beta_t in (0, 1).  This is one step of normalised LMS on
the in-context regression  v ~ S k  with forgetting alpha_t, i.e. a learned, input-gated
recursive least-squares / Kalman-type estimator of a *linear-in-parameters* map, whereas a
diagonal selective SSM (Mamba) can only exponentially average each channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, d_head=16, d_conv=4):
        super().__init__()
        self.h, self.dk, self.d_conv = n_heads, d_head, d_conv
        d_inner = n_heads * d_head
        self.d_inner = d_inner
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_inner, bias=False)
        self.conv = nn.Conv1d(3 * d_inner, 3 * d_inner, d_conv, groups=3 * d_inner, padding=d_conv - 1)
        self.ab = nn.Linear(d_model, 2 * n_heads)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, n_heads)))
        with torch.no_grad():  # start with slow forgetting (alpha ~ 0.97..0.99) and moderate beta
            self.ab.bias[:n_heads].fill_(-4.0)
            self.ab.bias[n_heads:].fill_(0.0)
        self.gate = nn.Linear(d_model, d_inner)
        self.onorm = nn.LayerNorm(d_head)
        self.out_proj = nn.Linear(d_inner, d_model)

    def _gates(self, xn):
        a, b = self.ab(xn).split(self.h, dim=-1)
        log_alpha = -F.softplus(a) * torch.exp(self.A_log)  # (.., H) <= 0
        return torch.exp(log_alpha), torch.sigmoid(b)

    def _qkv(self, c):
        q, k, v = c.split(self.d_inner, dim=-1)
        shp = c.shape[:-1] + (self.h, self.dk)
        q = F.normalize(F.silu(q).reshape(shp), dim=-1)
        k = F.normalize(F.silu(k).reshape(shp), dim=-1)
        return q, k, F.silu(v).reshape(shp)

    def _out(self, x, xn, o):
        o = self.onorm(o).reshape(*o.shape[:-2], self.d_inner)
        return x + self.out_proj(o * F.silu(self.gate(xn)))

    @staticmethod
    def _rec(S, q, k, v, alpha, beta):
        # S: (B,H,dv,dk); q,k,v: (B,H,d); alpha,beta: (B,H).  alpha*(S + beta (v - S k) k^T)
        Sk = torch.einsum('bhvk,bhk->bhv', S, k)
        S = alpha[..., None, None] * (S + torch.einsum('bhv,bhk->bhvk', beta[..., None] * (v - Sk), k))
        return S, torch.einsum('bhvk,bhk->bhv', S, q)

    def forward(self, x):
        B, T, _ = x.shape
        xn = self.norm(x)
        c = F.silu(self.conv(self.qkv(xn).transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = self._qkv(c)
        alpha, beta = self._gates(xn)
        S = x.new_zeros(B, self.h, self.dk, self.dk)
        outs = []
        for t in range(T):
            S, o = self._rec(S, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
            outs.append(o)
        return self._out(x, xn, torch.stack(outs, 1))

    def init_state(self, batch, device=None):
        device = device or self.A_log.device
        return (torch.zeros(batch, 3 * self.d_inner, self.d_conv - 1, device=device),
                torch.zeros(batch, self.h, self.dk, self.dk, device=device))

    def step(self, x, state):
        buf, S = state
        xn = self.norm(x)
        window = torch.cat([buf, self.qkv(xn)[..., None]], dim=-1)
        c = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        q, k, v = self._qkv(c)
        alpha, beta = self._gates(xn)
        S, o = self._rec(S, q, k, v, alpha, beta)
        return self._out(x, xn, o), (window[..., 1:], S)


class GatedDeltaStack(nn.Module):
    def __init__(self, d_in, d_out, d_model=64, n_layers=2, n_heads=4, d_head=16):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([GatedDeltaBlock(d_model, n_heads, d_head) for _ in range(n_layers)])
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
