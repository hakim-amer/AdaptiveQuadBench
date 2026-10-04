"""
Causal features and lumped-disturbance labels for the learned (selective-SSM) estimator.

The same FeatureExtractor runs offline on logged rollouts and online inside the controller,
so there is no train/deploy mismatch. Everything the controller sees is non-privileged:
(noisy) measured state, measured rotor speeds and the command it believes was applied.

Label definition (matches LumpedKF / NMPC parameter semantics exactly): over step [k, k+1]
    F_k   = m_nom * ((v_{k+1} - v_k)/dt - a_model(x_k, u_k))           (world frame)
    tau_k = J_nom @ ((w_{k+1} - w_k)/dt - wdot_model(x_k, u_k))         (body frame)
evaluated on the TRUE states, i.e. what a noise-free, zero-lag KF would report. At step k
the estimator must output label_k from information up to k (one-step-ahead prediction).
Targets / outputs are in acceleration units: [F/m, J^-1 tau].
"""

import numpy as np
import casadi as cs
from scipy.spatial.transform import Rotation

from icon_mpc.estimators import LumpedKF
from icon_mpc.nmpc import LAYOUT, build_model

# fast / medium / slow lumped KFs: fixed-gain operating points, none of which is best across
# regimes (see kf_grid results). The SSM learns to select / blend / correct them.
KF_BANK = (dict(q_F=30.0, q_tau=0.3, r_v=1e-3, r_w=1e-3),
           dict(q_F=10.0, q_tau=0.1, r_v=0.05, r_w=0.05),
           dict(q_F=10.0, q_tau=0.1, r_v=0.15, r_w=0.15),
           dict(q_F=1.0, q_tau=0.1, r_v=0.15, r_w=0.15))
BASE_KF = 2  # most robust fixed gain (100% success in every regime); output = base + correction
# outputs: 6 disturbance accels [F/m, J^-1 tau] (short-horizon mean) + 12 filtered state
# [p, v, w, attitude correction rotvec (world, left-multiplied onto the measured attitude)]
D_DIST, D_STATE = 6, 12
# feature layout: R(9) v(3) w(3) omega(4) u_prev(4) | residual(6) | increments(6) | KF bank(6 each)
RES_SLICE = slice(23, 29)
KF_START = 35
LABEL_AVG = 5  # disturbance label = mean of the next LABEL_AVG one-step labels


class _Model:
    def __init__(self, p_nom, k_eta, dt, tau_m):
        mdl = build_model('icon_feat_model')
        self.f = cs.Function('f', [mdl.x, mdl.u, mdl.p], [mdl.f_expl_expr])
        self.p = p_nom.copy()
        for k in ('F', 'tau', 'wind'):
            self.p[LAYOUT.slices[k]] = 0.0
        self.m = float(p_nom[LAYOUT.slices['m']][0])
        Jv = p_nom[LAYOUT.slices['J']]
        self.J = np.array([[Jv[0], Jv[3], Jv[4]], [Jv[3], Jv[1], Jv[5]], [Jv[4], Jv[5], Jv[2]]])
        self.Jinv = np.linalg.inv(self.J)
        self.k_eta, self.dt = k_eta, dt
        self.a1 = tau_m / dt * (1 - np.exp(-dt / tau_m)) if tau_m else None
        self.a2 = tau_m / (2 * dt) * (1 - np.exp(-2 * dt / tau_m)) if tau_m else None

    def thrust(self, omega0, omega_cmd):
        if self.a1 is None or omega_cmd is None:
            return self.k_eta * omega0 ** 2
        d = omega0 - omega_cmd
        return self.k_eta * (omega_cmd ** 2 + 2 * omega_cmd * d * self.a1 + d ** 2 * self.a2)

    def residual(self, x_prev, x, omega_prev, omega_cmd_prev):
        """[dv/dt - a_model, dw/dt - wdot_model] over one step (acceleration units)."""
        xdot = np.asarray(self.f(x_prev, self.thrust(omega_prev, omega_cmd_prev), self.p)).ravel()
        return np.concatenate([(x[3:6] - x_prev[3:6]) / self.dt - xdot[3:6],
                               (x[10:13] - x_prev[10:13]) / self.dt - xdot[10:13]])


def x13(state):
    q = state['q']
    return np.concatenate([state['x'], state['v'], [q[3], q[0], q[1], q[2]], state['w']])


class FeatureExtractor:
    def __init__(self, p_nom, k_eta, dt=0.01, tau_m=None):
        self.model = _Model(p_nom, k_eta, dt, tau_m)
        self.kfs = [LumpedKF(p_nom, k_eta, dt=dt, tau_m=tau_m, **kw) for kw in KF_BANK]
        self.f_hover = self.model.m * 9.81 / 4
        self.prev = None
        self.dim = 9 + 3 + 3 + 4 + 4 + 6 + 6 + 6 * len(KF_BANK)

    def kf_accel(self, kf):
        F, tau = kf.estimate()
        return np.concatenate([F / self.model.m, self.model.Jinv @ tau])

    def step(self, state, omega_cmd_prev=None):
        """Features at step k from the measured state_k and the rotor-speed command the
        controller believes was applied over [k-1, k]. Returns (features, base disturbance
        estimate, base state estimate [p_meas, v_kf, w_kf])."""
        x = x13(state)
        om = np.asarray(state['rotor_speeds'], float)
        for kf in self.kfs:
            kf.update(state, omega_cmd_prev)
        Rq = Rotation.from_quat(state['q'])
        if self.prev is None:
            res, inc = np.zeros(6), np.zeros(6)
        else:
            res = self.model.residual(self.prev[0], x, self.prev[1], omega_cmd_prev)
            # kinematic consistency: position / attitude increments vs measured v / w (noise cues)
            inc = np.concatenate([(x[0:3] - self.prev[0][0:3]) / self.model.dt - x[3:6],
                                  (self.prev[2].inv() * Rq).as_rotvec() / self.model.dt - x[10:13]])
        self.prev = (x, om.copy(), Rq)
        u_prev = (self.model.k_eta * omega_cmd_prev ** 2 / self.f_hover if omega_cmd_prev is not None
                  else np.ones(4))
        kfe = [self.kf_accel(kf) for kf in self.kfs]
        feat = np.concatenate([Rq.as_matrix().ravel(), x[3:6], x[10:13], om / 1000.0, u_prev, res, inc] + kfe)
        sbase = np.concatenate([x[0:3], self.kfs[BASE_KF].z[0:6], np.zeros(3)])
        return feat.astype(np.float32), kfe[BASE_KF].astype(np.float32), sbase.astype(np.float32)

    def to_wrench(self, acc6):
        """Acceleration-unit estimate -> (F_world [N], tau_body [Nm]) for the NMPC."""
        return self.model.m * acc6[:3], self.model.J @ acc6[3:]


def labels_from_truth(model, xs, omegas, omega_cmds, avg=LABEL_AVG, skip=()):
    """xs: (T, 13) true states, omegas: (T, 4) true rotor speeds, omega_cmds: (T, 4) applied
    commands over [k, k+1]. Returns (T-1, 6) labels in acceleration units: the mean one-step
    residual over [k, k+avg) (truncated at the end), i.e. the disturbance the NMPC will face next."""
    one = np.stack([model.residual(xs[k], xs[k + 1], omegas[k], omega_cmds[k])
                    for k in range(len(xs) - 1)])
    # state jumps (impacts) are not disturbances the NMPC can anticipate: replace those one-step
    # residuals by the last regular one
    for k in sorted(skip):
        if 0 <= k < len(one):
            one[k] = one[k - 1] if k > 0 else 0.0
    c = np.concatenate([np.zeros((1, 6)), np.cumsum(one, 0)])
    idx = np.arange(len(one))
    hi = np.minimum(idx + avg, len(one))
    return ((c[hi] - c[idx]) / (hi - idx)[:, None]).astype(np.float32)
