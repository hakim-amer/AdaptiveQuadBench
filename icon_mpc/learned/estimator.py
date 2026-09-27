"""
Online learned disturbance estimator: FeatureExtractor + selective SSM (recurrent step mode).

Drop-in replacement for LumpedKF inside OracleNMPC (level 'learned'): update(state, cmd)
returns (F_world, tau_body); .z exposes the base KF's filtered [v, w] for filt=1.
"""

import os

import numpy as np

from icon_mpc.learned.features import BASE_KF, FeatureExtractor

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'models')
_CACHE = {}


def load_model(name):
    if name not in _CACHE:
        import torch
        torch.set_num_threads(1)
        from icon_mpc.learned.train import build_model
        ck = torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu', weights_only=False)
        net = build_model(ck['cfg'])
        net.load_state_dict(ck['state_dict'])
        net.eval()
        _CACHE[name] = (net, ck)
    return _CACHE[name]


class LearnedEstimator:
    def __init__(self, p_nom, k_eta_ctrl, dt=0.01, tau_m=None, model='ssm', **_):
        import torch
        self.torch = torch
        self.fe = FeatureExtractor(p_nom, k_eta_ctrl, dt, tau_m)
        self.net, ck = load_model(model)
        self.mu, self.sd = (torch.as_tensor(ck[k]) for k in ('feat_mu', 'feat_sd'))
        self.out_sd = torch.as_tensor(ck['out_sd'])
        self.state = self.net.init_state(1, 'cpu')
        self.acc = np.zeros(6)
        self.xf = None  # learned filtered [p, v, w]

    @property
    def z(self):
        return self.fe.kfs[BASE_KF].z

    def update(self, state, omega_cmd_prev=None):
        feat, base, sbase = self.fe.step(state, omega_cmd_prev)
        with self.torch.no_grad():
            x = (self.torch.as_tensor(feat)[None] - self.mu) / self.sd
            y, self.state = self.net.step(x, self.state)
            corr = (y[0, :len(self.out_sd)] * self.out_sd).numpy()
        self.acc = base + corr[:6]
        self.xf = sbase + corr[6:15]
        return self.estimate()

    def estimate(self):
        return self.fe.to_wrench(self.acc)
