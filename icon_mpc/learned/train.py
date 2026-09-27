"""
Train the in-context disturbance estimator (selective SSM) on collected rollouts.

Target: normalised correction to the base KF, (label - base) / out_sd, with heteroscedastic
Gaussian NLL (the torque label carries unpredictable motor-noise; NLL lets the model say so).

    python -m icon_mpc.learned.train --data icon_mpc/results/data/v1 --name ssm_v1
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from icon_mpc.models.mamba import MambaStack

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models')


class GRUNet(nn.Module):
    """Ablation: non-selective (gated RNN) sequence model with the same step() interface."""

    def __init__(self, d_in, d_out, d_model=64, n_layers=2, **_):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.rnn = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_out))
        self.n_layers, self.d_model = n_layers, d_model

    def forward(self, x):
        return self.head(self.rnn(self.inp(x))[0])

    def init_state(self, batch, device=None):
        return torch.zeros(self.n_layers, batch, self.d_model, device=device)

    def step(self, x, h):
        y, h = self.rnn(self.inp(x)[:, None], h)
        return self.head(y[:, 0]), h


def build_model(cfg):
    if cfg['arch'] == 'gru':
        return GRUNet(cfg['d_in'], cfg['d_out'], cfg['d_model'], cfg['n_layers'])
    return MambaStack(cfg['d_in'], cfg['d_out'], cfg['d_model'], cfg['n_layers'], cfg['d_state'],
                      selective=cfg['arch'] != 'lti')


def load_data(paths):
    ds = [np.load(os.path.join(p, 'data.npz')) for p in paths]
    cat = lambda k: np.concatenate([d[k] for d in ds])
    return {k: cat(k) for k in ('feat', 'base', 'label', 'sbase', 'slabel', 'regime', 'experiment', 'trial')}


def per_regime_rmse(err, regimes):
    """err: (N, T, 6) in acceleration units -> {regime: (force_rmse, torque_rmse)}."""
    out = {}
    for r in np.unique(regimes):
        e = err[regimes == r]
        out[r] = (float(np.sqrt((e[..., :3] ** 2).mean())), float(np.sqrt((e[..., 3:] ** 2).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--arch', default='mamba', choices=['mamba', 'lti', 'gru'])
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--d_state', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    D = load_data(args.data)
    N = len(D['feat'])
    # split by trial index so a validation trial is unseen in every regime/experiment
    trials = np.unique(D['trial'])
    val_trials = np.random.default_rng(0).choice(trials, max(1, int(len(trials) * args.val_frac)), replace=False)
    is_val = np.isin(D['trial'], val_trials)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]

    feat = torch.as_tensor(D['feat'])
    # 15-dim target: [disturbance accel (6), state p, v, w (9)], each as a correction to its base
    base = torch.cat([torch.as_tensor(D['base']), torch.as_tensor(D['sbase'])], -1)
    label = torch.cat([torch.as_tensor(D['label']), torch.as_tensor(D['slabel'])], -1)
    DO = label.shape[-1]
    mu = feat[tr].reshape(-1, feat.shape[-1]).mean(0)
    sd = feat[tr].reshape(-1, feat.shape[-1]).std(0).clamp(min=1e-4)
    target = label - base
    out_sd = target[tr].reshape(-1, DO).std(0).clamp(min=1e-5)
    X = ((feat - mu) / sd).to(dev)
    Y = (target / out_sd).to(dev)
    print(f'{N} sequences ({len(tr)} train / {len(va)} val), T={feat.shape[1]}, d_in={feat.shape[-1]}')
    print('out_sd', out_sd.numpy().round(4))

    cfg = dict(arch=args.arch, d_in=feat.shape[-1], d_out=2 * DO, d_model=args.d_model,
               n_layers=args.n_layers, d_state=args.d_state)
    net = build_model(cfg).to(dev)
    print(f'{sum(p.numel() for p in net.parameters())} params')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)

    def loss_fn(out, y):
        mean, logvar = out[..., :DO], out[..., DO:].clamp(-8, 6)
        return (0.5 * ((mean - y) ** 2 * torch.exp(-logvar) + logvar)).mean(), ((mean - y) ** 2).mean()

    def evaluate(idx):
        net.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                preds.append(net(X[idx[s:s + 64]])[..., :DO].cpu())
        net.train()
        pred = torch.cat(preds) * out_sd + base[idx]
        return (pred - label[idx]).numpy()

    t0, best, best_state = time.time(), np.inf, None
    for ep in range(args.epochs):
        perm = np.random.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm) - args.bs + 1, args.bs):
            b = perm[s:s + args.bs]
            nll, mse = loss_fn(net(X[b]), Y[b])
            opt.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += mse.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            err = evaluate(va)
            v = float(np.sqrt((err ** 2).mean()))
            if v < best:
                best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            print(f'ep {ep + 1:3d}  train_mse {tot / max(1, len(perm) // args.bs):.4f}  val_rmse {v:.4f}  '
                  f'({time.time() - t0:.0f}s)', flush=True)

    net.load_state_dict(best_state)
    err = evaluate(va)
    reg = D['regime'][va]
    print('\nval RMSE per regime  [force m/s^2 / torque rad/s^2]    state [p mm / v cm/s / w cm/s]:')
    lab6 = label[va][..., :6]
    kf_err = {j: (feat[va][..., -18 + 6 * j:-12 + 6 * j if j < 2 else None] - lab6).numpy()
              for j in range(3)}
    rows = {'learned': per_regime_rmse(err[..., :6], reg)}
    rows.update({f'kf{j}': per_regime_rmse(kf_err[j], reg) for j in range(3)})
    serr_l = err[..., 6:]
    serr_b = (base[va][..., 6:] - label[va][..., 6:]).numpy()
    srm = lambda e, m: (1e3 * np.sqrt((e[m][..., 0:3] ** 2).mean()), 1e2 * np.sqrt((e[m][..., 3:6] ** 2).mean()),
                        1e2 * np.sqrt((e[m][..., 6:9] ** 2).mean()))
    for r in np.unique(reg):
        m = reg == r
        print(f'  {r:12s} ' + '  '.join(f'{n}: {rows[n][r][0]:.3f}/{rows[n][r][1]:.3f}' for n in rows)
              + '   state learned %.1f/%.1f/%.1f' % srm(serr_l, m) + '  meas+kf %.1f/%.1f/%.1f' % srm(serr_b, m))
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({'cfg': cfg, 'state_dict': best_state, 'feat_mu': mu, 'feat_sd': sd, 'out_sd': out_sd,
                'val_trials': val_trials, 'args': vars(args)}, os.path.join(MODEL_DIR, f'{args.name}.pt'))
    print('saved', os.path.join(MODEL_DIR, f'{args.name}.pt'))


if __name__ == '__main__':
    main()
