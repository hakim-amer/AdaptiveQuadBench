"""
Non-privileged estimators that fill the NMPC's disturbance parameter slots.

LumpedKF: linear time-varying Kalman filter on z = [v(3), w(3), F(3, world), tau(3, body)].
The known part of the dynamics is the NMPC prediction model itself (nominal params + rotor
drag, zero wind) driven by the *measured* rotor speeds, so whatever that model misses
(wind, payload, parameter mismatch, rotor faults, external wrenches) is absorbed into F / tau,
which are modelled as random walks and passed to the NMPC exactly like the oracle's F / tau.
"""

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
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.z = z_pred + K @ (y - self.H @ z_pred)
        self.P = (np.eye(12) - K @ self.H) @ P_pred
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
