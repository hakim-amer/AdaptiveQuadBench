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

    def _gates(self, xn, log=False):
        a, b = self.ab(xn).split(self.h, dim=-1)
        log_alpha = -F.softplus(a) * torch.exp(self.A_log)  # (.., H) <= 0
        return (log_alpha if log else torch.exp(log_alpha)), torch.sigmoid(b)

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

    chunk = 64

    def forward(self, x, sequential=False):
        B, T, _ = x.shape
        xn = self.norm(x)
        c = F.silu(self.conv(self.qkv(xn).transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = self._qkv(c)
        if sequential:  # reference implementation (one recurrence step per time step)
            alpha, beta = self._gates(xn)
            S = x.new_zeros(B, self.h, self.dk, self.dk)
            outs = []
            for t in range(T):
                S, o = self._rec(S, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
                outs.append(o)
            return self._out(x, xn, torch.stack(outs, 1))
        log_alpha, beta = self._gates(xn, log=True)
        return self._out(x, xn, self._chunked(q, k, v, log_alpha, beta))

    def _chunked(self, q, k, v, log_alpha, beta):
        """Exact chunk-parallel form of S_t = a_t (S_{t-1} + b_t (v_t - S_{t-1} k_t) k_t^T).

        Within a chunk with start state S0 and G_t = sum_{i<=t} log a_i (G_0 = 0):
            w_i + b_i sum_{j<i} e^{G_{i-1}-G_{j-1}} (k_i.k_j) w_j = b_i (v_i - e^{G_{i-1}} S0 k_i)
            o_t = e^{G_t} S0 q_t + sum_{i<=t} e^{G_t-G_{i-1}} (k_i.q_t) w_i
            S_C = e^{G_C} S0 + sum_i e^{G_C-G_{i-1}} w_i k_i^T
        (all decay ratios <= 1). One unit-lower-triangular solve per chunk replaces C steps."""
        B, T, H, d = q.shape
        C = min(self.chunk, T)
        pad = (-T) % C
        if pad:  # identity steps: a = 1, b = 0
            q, k, v = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (q, k, v))
            log_alpha, beta = F.pad(log_alpha, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad))
        n = (T + pad) // C
        # (B, H, n, C, d) / (B, H, n, C)
        q, k, v = (z.reshape(B, n, C, H, d).permute(0, 3, 1, 2, 4) for z in (q, k, v))
        la = log_alpha.reshape(B, n, C, H).permute(0, 3, 1, 2)
        b = beta.reshape(B, n, C, H).permute(0, 3, 1, 2)
        G = la.cumsum(-1)                       # G_t
        Gp = G - la                             # G_{t-1}
        idx = torch.arange(C, device=q.device)
        strict = idx[:, None] > idx[None, :]
        causal = idx[:, None] >= idx[None, :]
        # A_ij = b_i e^{G_{i-1} - G_{j-1}} k_i.k_j  (j < i)
        dA = (Gp[..., :, None] - Gp[..., None, :]).masked_fill(~strict, -float('inf'))
        A = b[..., None] * torch.exp(dA) * (k @ k.transpose(-1, -2))
        L = torch.eye(C, device=q.device, dtype=q.dtype) + A
        # decays for outputs: e^{G_t - G_{i-1}}, i <= t
        dO = (G[..., :, None] - Gp[..., None, :]).masked_fill(~causal, -float('inf'))
        Dqk = torch.exp(dO) * (q @ k.transpose(-1, -2))          # (B,H,n,C,C)
        eG, eGp = torch.exp(G)[..., None], torch.exp(Gp)[..., None]
        eGC = torch.exp(G[..., -1:, None] - Gp[..., :, None])     # e^{G_C - G_{i-1}}
        S = q.new_zeros(B, H, d, d)
        outs = []
        for j in range(n):
            rhs = b[:, :, j, :, None] * (v[:, :, j] - eGp[:, :, j] * (k[:, :, j] @ S.transpose(-1, -2)))
            w = torch.linalg.solve_triangular(L[:, :, j], rhs, upper=False, unitriangular=True)
            o = eG[:, :, j] * (q[:, :, j] @ S.transpose(-1, -2)) + Dqk[:, :, j] @ w
            outs.append(o)
            S = torch.exp(G[:, :, j, -1])[..., None, None] * S + (eGC[:, :, j] * w).transpose(-1, -2) @ k[:, :, j]
        o = torch.stack(outs, 2)                                  # (B,H,n,C,d)
        return o.permute(0, 2, 3, 1, 4).reshape(B, n * C, H, d)[:, :T]

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
