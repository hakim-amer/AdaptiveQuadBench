"""
Non-privileged estimators that fill the NMPC's disturbance parameter slots.

LumpedKF: linear time-varying Kalman filter on z = [v(3), w(3), F(3, world), tau(3, body)].
The known part of the dynamics is the NMPC prediction model itself (nominal params + rotor
drag, zero wind) driven by the *measured* rotor speeds, so whatever that model misses
(wind, payload, parameter mismatch, rotor faults, external wrenches) is absorbed into F / tau,
which are modelled as random walks and passed to the NMPC exactly like the oracle's F / tau.
"""

from collections import deque

import numpy as np
import casadi as cs

from icon_mpc.nmpc import LAYOUT, build_model


class LumpedKF:
    def __init__(self, p_nom, k_eta_ctrl, dt=0.01, q_v=0.05, q_w=0.5, q_F=10.0, q_tau=0.3,
                 r_v=1e-3, r_w=1e-2, tau_m=None):
        """q_* are continuous-time process-noise PSD roots (units/s/sqrt(Hz)); r_* measurement std."""
        model = build_model('icon_kf_model')
        self._f = cs.Function('f', [model.x, model.u, model.p], [model.f_expl_expr])
        self.p = p_nom.copy()
        self.p[LAYOUT.slices['F']] = 0.0
        self.p[LAYOUT.slices['tau']] = 0.0
        self.p[LAYOUT.slices['wind']] = 0.0
        self.m = float(p_nom[LAYOUT.slices['m']][0])
        Jv = p_nom[LAYOUT.slices['J']]
        J = np.array([[Jv[0], Jv[3], Jv[4]], [Jv[3], Jv[1], Jv[5]], [Jv[4], Jv[5], Jv[2]]])
        self.Jinv = np.linalg.inv(J)
        self.k_eta, self.dt = k_eta_ctrl, dt
        # First-order motor omega(s) = c + (omega0 - c) exp(-s/tau_m): exact step-averages of
        # omega and omega^2 over [0, dt] (tau_m=None -> treat measured speed as held).
        if tau_m:
            self.a1 = tau_m / dt * (1 - np.exp(-dt / tau_m))
            self.a2 = tau_m / (2 * dt) * (1 - np.exp(-2 * dt / tau_m))
        else:
            self.a1 = self.a2 = None

        self.z = np.zeros(12)
        self.P = np.diag([1e-4] * 6 + [1.0] * 3 + [1e-2] * 3)
        self.Q = np.diag(np.concatenate([np.full(3, q_v ** 2), np.full(3, q_w ** 2),
                                         np.full(3, q_F ** 2), np.full(3, q_tau ** 2)]) * dt)
        self.R = np.diag(np.concatenate([np.full(3, r_v ** 2), np.full(3, r_w ** 2)]))
        self.H = np.hstack([np.eye(6), np.zeros((6, 6))])
        self.prev = None

    @staticmethod
    def _x13(state):
        q = state['q']
        return np.concatenate([state['x'], state['v'], [q[3], q[0], q[1], q[2]], state['w']])

    def _thrust_input(self, omega0, omega_cmd):
        if self.a1 is None or omega_cmd is None:
            return self.k_eta * omega0 ** 2
        d = omega0 - omega_cmd
        return self.k_eta * (omega_cmd ** 2 + 2 * omega_cmd * d * self.a1 + d ** 2 * self.a2)

    def update(self, state, omega_cmd_prev=None):
        """Call once per control step with the (non-privileged) measured state and the rotor
        speed command that was applied over the previous step."""
        x = self._x13(state)
        y = np.concatenate([x[3:6], x[10:13]])
        if self.prev is None:
            self.z[:6] = y
            self.prev = (x, np.asarray(state['rotor_speeds'], float).copy())
            return self.estimate()
        x_prev, omega_prev = self.prev
        u = self._thrust_input(omega_prev, omega_cmd_prev)
        xdot = np.asarray(self._f(x_prev, u, self.p)).ravel()
        dt = self.dt

        # predict: v+ = v + dt(a_model + F/m); w+ = w + dt(wdot_model + Jinv tau)
        A = np.eye(12)
        A[0:3, 6:9] = dt / self.m * np.eye(3)
        A[3:6, 9:12] = dt * self.Jinv
        z_pred = A @ self.z
        # replace the (v, w) propagation by the model evaluated at the measured previous state
        z_pred[0:3] = x_prev[3:6] + dt * (xdot[3:6] + self.z[6:9] / self.m)
        z_pred[3:6] = x_prev[10:13] + dt * (xdot[10:13] + self.Jinv @ self.z[9:12])
        P_pred = A @ self.P @ A.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R
        Sinv = np.linalg.inv(S)
        K = P_pred @ self.H.T @ Sinv
        nu = y - self.H @ z_pred
        # innovation log-likelihood (used by the multiple-model estimator)
        self.loglik = -0.5 * (nu @ Sinv @ nu + np.linalg.slogdet(S)[1])
        self.z = z_pred + K @ nu
        self.P = (np.eye(12) - K @ self.H) @ P_pred
        self.a_model = xdot[3:6].copy()  # model translational accel (zero disturbance) at x_prev
        self.prev = (x, np.asarray(state['rotor_speeds'], float).copy())
        return self.estimate()

    def estimate(self):
        return self.z[6:9].copy(), self.z[9:12].copy()


class DelayID:
    """Classical actuation-latency identifier (multiple-hypothesis, rotor-speed based).

    For each candidate delay d (in steps) predict the measured rotor speeds with the first-order
    motor model driven by the command issued d steps earlier, and keep an exponentially-forgotten
    squared prediction error; the estimate is the arg-min hypothesis. Needs rotor-speed feedback.
    """

    def __init__(self, dt, tau_m, max_steps=8, forget=0.98):
        self.a = float(np.exp(-dt / tau_m)) if tau_m else 0.0
        self.cost = np.zeros(max_steps + 1)
        self.max_steps, self.forget = max_steps, forget
        self.prev = None
        self.d = 0

    def update(self, omega_meas, cmd_speed_hist):
        """omega_meas: measured rotor speeds now; cmd_speed_hist: issued rotor-speed commands,
        newest last (the newest was issued one step ago). Returns the delay estimate in steps."""
        om = np.asarray(omega_meas, float)
        if self.prev is not None:
            n = len(cmd_speed_hist)
            for d in range(self.max_steps + 1):
                c = cmd_speed_hist[max(-n, -1 - d)]
                pred = c + (self.prev - c) * self.a
                self.cost[d] = self.forget * self.cost[d] + float(np.sum((om - pred) ** 2))
            self.d = int(np.argmin(self.cost))
        self.prev = om
        return self.d


class MMAE:
    """Multiple-model adaptive estimator (classical adaptive baseline to the learned blend).

    Runs a bank of fixed-gain LumpedKFs and mixes their estimates with posterior model
    probabilities from exponentially-forgotten innovation log-likelihoods (forgetting lets the
    posterior switch when the regime changes). A probability floor keeps every model alive.
    """

    def __init__(self, p_nom, k_eta_ctrl, dt=0.01, tau_m=None, bank=None, forget=0.98, floor=1e-3, **_):
        from icon_mpc.learned.features import KF_BANK
        self.kfs = [LumpedKF(p_nom, k_eta_ctrl, dt=dt, tau_m=tau_m, **kw) for kw in (bank or KF_BANK)]
        self.L = np.zeros(len(self.kfs))
        self.forget, self.floor = forget, floor
        self.w = np.full(len(self.kfs), 1.0 / len(self.kfs))

    def update(self, state, omega_cmd_prev=None):
        for i, kf in enumerate(self.kfs):
            kf.update(state, omega_cmd_prev)
            self.L[i] = self.forget * self.L[i] + getattr(kf, 'loglik', 0.0)
        w = np.exp(self.L - self.L.max())
        w = w / w.sum()
        w = np.maximum(w, self.floor)
        self.w = w / w.sum()
        return self.estimate()

    @property
    def z(self):
        return sum(w * kf.z for w, kf in zip(self.w, self.kfs))

    def estimate(self):
        F = sum(w * kf.z[6:9] for w, kf in zip(self.w, self.kfs))
        tau = sum(w * kf.z[9:12] for w, kf in zip(self.w, self.kfs))
        return F, tau


class RobustFrontEnd:
    """Measurement front end in front of every estimator / the NMPC (no learning).

    Sets .event each step:
      'stale'  : packet bit-identical to the previous one (impossible with live sensor noise) ->
                 missing data; estimators must skip their update and the NMPC dead-reckons;
      'resume' : first fresh packet after a stale burst -> estimators re-initialise kinematics;
      'jump'   : a genuine state jump (impact): two consecutive packets agree with each other but
                 not with the prediction -> accepted after one step, estimators re-initialise
                 kinematics instead of attributing the jump to a (huge) force;
      None     : normal. Isolated innovations > k x running robust scale are rejected as outliers
                 (constant-velocity / constant-rate prediction used instead).
    """

    FLOOR = {'x': 2e-3, 'v': 2e-2, 'att': 5e-3, 'w': 2e-2, 'rotor_speeds': 10.0}
    KEY = {'x': 'x', 'v': 'v', 'att': 'q', 'w': 'w', 'rotor_speeds': 'rotor_speeds'}

    def __init__(self, dt, k=6.0, warmup=20, max_rej=5, grace=10):
        self.dt, self.k, self.warmup, self.max_rej, self.grace = dt, k, warmup, max_rej, grace
        self.grace_left = 0
        self.out = self.raw = None
        self.scale = {g: 0.0 for g in self.FLOOR}
        self.rej = {g: 0 for g in self.FLOOR}
        self.last_rej = {g: None for g in self.FLOOR}
        self.n, self.event, self.was_stale = 0, None, False
        self.n_reject = self.n_stale = self.n_jump = 0

    def _predict(self):
        from scipy.spatial.transform import Rotation
        o, dt = self.out, self.dt
        return {'x': o['x'] + dt * o['v'], 'v': o['v'].copy(),
                'q': (Rotation.from_quat(o['q']) * Rotation.from_rotvec(dt * o['w'])).as_quat(),
                'w': o['w'].copy(), 'rotor_speeds': o['rotor_speeds'].copy()}

    @staticmethod
    def _dist(g, a, b):
        from scipy.spatial.transform import Rotation
        if g == 'att':
            return np.linalg.norm((Rotation.from_quat(a) * Rotation.from_quat(b).inv()).as_rotvec())
        return np.linalg.norm(a - b)

    def filter(self, state, pred=None):
        """pred: optional model-based one-step prediction {'x','v','q','w'} of the previous output."""
        meas = {k: np.array(state[k], float) for k in ('x', 'v', 'q', 'w', 'rotor_speeds')}
        self.event = None
        if self.out is None:
            self.out, self.raw = meas, meas
            return dict(meas)
        stale = all(np.array_equal(meas[k], self.raw[k]) for k in ('x', 'v', 'q', 'w'))
        self.raw = meas
        if stale:
            self.n_stale += 1
            self.event, self.was_stale = 'stale', True
            self.out = dict(self._predict(), **(pred or {}))
            return dict(self.out)
        if self.was_stale:  # accept everything after a blackout
            self.was_stale = False
            self.event = 'resume'
            self.out = meas
            self.rej = {g: 0 for g in self.FLOOR}
            return dict(meas)
        self.n += 1
        if self.grace_left > 0:  # post-jump transient: the predictor is not yet trustworthy
            self.grace_left -= 1
            self.out = meas
            return dict(meas)
        pred = dict(self._predict(), **(pred or {}))
        out = {}
        for g, mk in self.KEY.items():
            r = self._dist(g, meas[mk], pred[mk])
            thr = self.k * self.scale[g] + self.FLOOR[g]
            if self.n > self.warmup and r > thr:
                lr = self.last_rej[g]
                if lr is not None:  # propagate the previously rejected packet one step with its own rates
                    if g == 'x':
                        lr = lr + self.dt * meas['v']
                    elif g == 'att':
                        from scipy.spatial.transform import Rotation
                        lr = (Rotation.from_quat(lr) * Rotation.from_rotvec(self.dt * meas['w'])).as_quat()
                consistent = lr is not None and self._dist(g, meas[mk], lr) < thr
                if consistent or self.rej[g] >= self.max_rej:
                    out[mk] = meas[mk]  # genuine jump
                    if g in ('v', 'w', 'att', 'x'):
                        self.event = 'jump'
                    self.n_jump += 1
                    self.rej[g], self.last_rej[g] = 0, None
                else:
                    out[mk] = pred[mk]
                    self.rej[g] += 1
                    self.last_rej[g] = meas[mk]
                    self.n_reject += 1
                continue
            out[mk] = meas[mk]
            self.rej[g], self.last_rej[g] = 0, None
            self.scale[g] = (0.98 * self.scale[g] + 0.02 * r) if self.n > 1 else r
        if self.event == 'jump':
            self.grace_left = self.grace
            self.rej = {g: 0 for g in self.FLOOR}
            self.last_rej = {g: None for g in self.FLOOR}
            out = meas  # accept the whole consistent packet
        self.out = out
        return dict(out)


def reset_kinematics(est):
    """Re-initialise an estimator's kinematic memory (after missing data or a state jump)
    while keeping its disturbance estimate."""
    if est is None:
        return
    if hasattr(est, 'experts'):
        for e in est.experts:
            reset_kinematics(e)
        est.hist.clear()
    if hasattr(est, 'kfs'):
        for kf in est.kfs:
            kf.prev = None
    if hasattr(est, 'fe'):
        reset_kinematics(est.fe)
    if hasattr(est, 'prev'):
        est.prev = None


class SafeAggregator:
    """Online exponentially-weighted aggregation of disturbance estimators (guarantee G4).

    Experts predict the force disturbance *before* the data that scores them arrives. The score is
    the observable low-frequency prediction loss of the position second difference over k steps
    (a 2k-step triangular average of acceleration), which is unbiased for the expert's
    low-frequency force error (the control-relevant part, guarantee G2) up to expert-independent
    noise. Losses are Huber-clipped and exponentially forgotten (tracking the best expert in a
    switching environment); eta is set scale-free from the running loss level.
    """

    def __init__(self, experts, mass, dt, k=20, forget=0.995, floor=0.02, huber=4.0):
        self.experts, self.m, self.dt, self.k = experts, mass, dt, k
        self.forget, self.floor, self.huber = forget, floor, huber
        n = len(experts)
        self.L = np.zeros(n)
        self.w = np.full(n, 1.0 / n)
        self.hist = deque(maxlen=2 * k + 1)  # (p_meas, a_model, [F_j/m])
        self.lscale = None
        tri = np.concatenate([np.arange(1, k + 1), np.arange(k - 1, 0, -1)]).astype(float)
        self.tri = tri / tri.sum()
        self.w_log = []

    def update(self, state, omega_cmd_prev=None):
        ests = [e.update(state, omega_cmd_prev) for e in self.experts]
        base = self.experts[0]
        a_model = getattr(base, 'a_model', None)
        if a_model is None and hasattr(base, 'fe'):
            a_model = getattr(base.fe.kfs[0], 'a_model', None)
        # store predictions made with information up to the previous step
        prevF = [np.asarray(getattr(e, '_lastF', est[0])) for e, est in zip(self.experts, ests)]
        if a_model is not None:
            self.hist.append((np.array(state['x'], float), a_model.copy(), [f / self.m for f in prevF]))
        for e, est in zip(self.experts, ests):
            e._lastF = np.asarray(est[0], float).copy()
        if len(self.hist) == self.hist.maxlen:
            p = [h[0] for h in self.hist]
            a_meas = (p[-1] - 2 * p[self.k] + p[0]) / (self.k * self.dt) ** 2
            # p_2k - 2 p_k + p_0 ~ dt^2 * sum of triangular-weighted (1..k..1) accelerations of the
            # 2k-1 inner intervals; weights sum to k^2, so a_meas is their weighted mean
            inner = list(self.hist)[1:-1]
            am = np.array([h[1] for h in inner])
            loss = np.empty(len(self.experts))
            for j in range(len(self.experts)):
                dj = np.array([h[2][j] for h in inner])
                pred = self.tri @ (am + dj)
                loss[j] = np.sum((a_meas - pred) ** 2)
            lvl = np.min(loss)
            self.lscale = lvl if self.lscale is None else 0.99 * self.lscale + 0.01 * lvl
            c = self.huber * max(self.lscale, 1e-6)
            loss = np.where(loss > c, 2 * np.sqrt(loss * c) - c, loss)
            self.L = self.forget * self.L + loss
            eta = 1.0 / (2.0 * max(self.lscale, 1e-6) * 10.0)
            w = np.exp(-eta * (self.L - self.L.min()))
            w = np.maximum(w / w.sum(), self.floor)
            self.w = w / w.sum()
        self.w_log.append(self.w.copy())
        return self.estimate()

    @property
    def z(self):
        return sum(w * e.z for w, e in zip(self.w, self.experts))

    def estimate(self):
        Fs, taus = zip(*[e.estimate() for e in self.experts])
        return sum(w * f for w, f in zip(self.w, Fs)), sum(w * t for w, t in zip(self.w, taus))
