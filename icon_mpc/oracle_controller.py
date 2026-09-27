"""
Oracle NMPC controllers for the headroom study.

Levels (cumulative):
    nominal : controller's own (possibly wrong) params, no aero, no disturbance
    aero    : + rotor-drag structure (what offline residual learning could capture)
    params  : + ground-truth *current* vehicle params (mass/inertia/CoM/rotor gains)
    dist    : + ground-truth *current* external force/torque/wind, held over horizon
    future  : + ground-truth *future* params & disturbances at every horizon stage
"""

import copy
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from icon_mpc.nmpc import ParamNMPCSolver, get_param_nmpc, LAYOUT, nominal_params, vehicle_params, GRAV

LEVELS = ['nominal', 'aero', 'kf', 'learned', 'params', 'dist', 'future']  # levels >= 'params' are privileged


def flat_to_state_ref(flat, mass, thrust_gain_sum):
    """Differential-flatness reference: returns (x_ref(13), u_ref(4))."""
    a, j = flat['x_ddot'], flat['x_dddot']
    yaw, yaw_dot = flat['yaw'], flat['yaw_dot']
    t_vec = a + np.array([0, 0, GRAV])
    c = np.linalg.norm(t_vec)
    zb = t_vec / c
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    Rm = np.column_stack([xb, yb, zb])
    hw = (j - np.dot(zb, j) * zb) / c
    w = np.array([-np.dot(hw, yb), np.dot(hw, xb), yaw_dot * zb[2]])
    qx, qy, qz, qw = Rotation.from_matrix(Rm).as_quat()
    x_ref = np.concatenate([flat['x'], flat['x_dot'], [qw, qx, qy, qz], w])
    u_ref = np.full(4, mass * c / thrust_gain_sum)
    return x_ref, u_ref


class Privileged:
    """Ground-truth handles given only to oracle controllers."""

    def __init__(self, vehicle, wind_seq=None, ext_force=None, ext_torque=None,
                 toggle_times=None, dt=0.01):
        self.vehicle = vehicle
        self.wind_seq = wind_seq
        self.dt = dt
        self.ext_force = np.zeros(3) if ext_force is None else np.asarray(ext_force, float)
        self.ext_torque = np.zeros(3) if ext_torque is None else np.asarray(ext_torque, float)
        self.toggle_times = list(toggle_times) if toggle_times else []
        self.p_mass = float(getattr(vehicle, 'payload_mass', 0.0))
        self.p_pos = np.asarray(getattr(vehicle, 'payload_position', np.zeros(3)), float).copy()
        # Pre-compute detached / attached ground-truth vehicle copies
        self.veh_off = copy.deepcopy(vehicle)
        self.veh_on = copy.deepcopy(vehicle)
        self.payload_torque = np.zeros(3)
        if self.p_mass > 0:
            self.veh_on.attach_payload()
            r = self.veh_on.payload_position - self.veh_on.com
            self.payload_torque = np.cross(r, np.array([0, 0, -self.p_mass * self.veh_on.g]))

    def _on(self, t):
        if not self.toggle_times:
            return None
        return sum(1 for tt in self.toggle_times if t >= tt) % 2 == 1

    def wind_at(self, t):
        if self.wind_seq is None:
            return np.zeros(3)
        i = int(round(t / self.dt))
        return self.wind_seq[min(max(i, 0), len(self.wind_seq) - 1)]

    def truth_at(self, t, k_eta_ctrl):
        """Ground-truth parameter vector for the interval ending at time t."""
        on = self._on(t)
        if on is None:
            veh, F, tau = self.veh_off, self.ext_force, self.ext_torque
        else:
            veh = self.veh_on if (on and self.p_mass > 0) else self.veh_off
            F = self.ext_force if on else np.zeros(3)
            tau = (self.ext_torque + (self.payload_torque if self.p_mass > 0 else 0)) if on else np.zeros(3)
        p = vehicle_params(veh, k_eta_ctrl)
        p[LAYOUT.slices['F']] = F
        p[LAYOUT.slices['tau']] = tau
        p[LAYOUT.slices['wind']] = self.wind_at(t)
        return p


class OracleNMPC(MultirotorControlTemplate):
    def __init__(self, ctrl_params, level='nominal', privileged=None, solve_every=1,
                 t_horizon=0.5, n_nodes=10, sim_dt=0.01, dist_lag=None, kf_kwargs=None):
        super().__init__(ctrl_params)
        # dist_lag (s): 'dist' level sees the ground truth from dist_lag ago (emulates estimator delay)
        self.dist_lag = dist_lag
        assert level in LEVELS
        if LEVELS.index(level) >= LEVELS.index('params'):
            assert privileged is not None, f'level {level} needs privileged info'
        self.level, self.priv = level, privileged
        self.solve_every, self.sim_dt = solve_every, sim_dt
        self.N, self.T = n_nodes, t_horizon
        self.k_eta_ctrl = ctrl_params['k_eta']
        f_max = self.k_eta_ctrl * ctrl_params['rotor_speed_max'] ** 2
        self.mpc = get_param_nmpc(f_max, t_horizon, n_nodes)
        kf_kwargs = dict(kf_kwargs or {})
        self.record = None  # set to [] to log (measured state, believed applied command) per step
        aero = bool(kf_kwargs.pop('aero', level != 'nominal'))
        # delay (s): known actuation latency -> KF uses the command actually applied and the NMPC
        # predicts the state forward over the delay; filt: feed KF-filtered v, w to the NMPC.
        self.delay_steps = int(round(kf_kwargs.pop('delay', 0.0) / sim_dt))
        self.filt = int(kf_kwargs.pop('filt', 0))
        self.cmd_hist = deque(maxlen=self.delay_steps + 1)
        self._f = None
        if self.delay_steps:
            from icon_mpc.nmpc import build_model
            import casadi as cs
            mdl = build_model('icon_pred_model')
            self._f = cs.Function('f', [mdl.x, mdl.u, mdl.p], [mdl.f_expl_expr])
        self.p_nom = nominal_params(ctrl_params, self.k_eta_ctrl, aero=aero)
        self.trajectory = None
        self.step = 0
        self.u = np.full(4, ctrl_params['mass'] * GRAV / 4)
        self.solve_times = []
        self.kf = None
        if level == 'kf':
            from icon_mpc.estimators import LumpedKF
            kw = dict(tau_m=ctrl_params.get('tau_m'))
            kw.update(kf_kwargs)
            self.kf = LumpedKF(self.p_nom, self.k_eta_ctrl, dt=sim_dt, **kw)
        elif level == 'learned':
            from icon_mpc.learned.estimator import LearnedEstimator
            self.kf = LearnedEstimator(self.p_nom, self.k_eta_ctrl, dt=sim_dt,
                                       tau_m=ctrl_params.get('tau_m'), **kf_kwargs)
        self.est_log = []

    def update_trajectory(self, trajectory):
        self.trajectory = trajectory
        self.step = 0

    def _stage_params(self, t, state):
        N = self.N
        if self.level in ('nominal', 'aero'):
            return np.tile(self.p_nom, (N + 1, 1))
        if self.level in ('kf', 'learned'):
            p = self.p_nom.copy()
            p[LAYOUT.slices['F']], p[LAYOUT.slices['tau']] = self.kf.estimate()
            return np.tile(p, (N + 1, 1))
        if self.level == 'future':
            ts = t + self.sim_dt + np.arange(N + 1) * (self.T / N)
            return np.stack([self.priv.truth_at(tk, self.k_eta_ctrl) for tk in ts])
        if self.level == 'dist' and self.dist_lag is not None:
            return np.tile(self.priv.truth_at(max(t - self.dist_lag, 0.0), self.k_eta_ctrl), (N + 1, 1))
        p = vehicle_params(self.priv.vehicle, self.k_eta_ctrl)
        if self.level == 'dist':
            p[LAYOUT.slices['F']] = state.get('ext_force', np.zeros(3))
            p[LAYOUT.slices['tau']] = state.get('ext_torque', np.zeros(3))
            p[LAYOUT.slices['wind']] = state.get('wind', np.zeros(3))
        return np.tile(p, (N + 1, 1))

    def _predict(self, x, params):
        """RK4 roll-forward over the actuation delay with the commands already in the pipeline."""
        pend = list(self.cmd_hist)[-self.delay_steps:] if self.cmd_hist else []
        pend = [pend[0]] * (self.delay_steps - len(pend)) + pend if pend else [self.u] * self.delay_steps
        f = lambda xx, uu: np.asarray(self._f(xx, uu, params)).ravel()
        h = self.sim_dt
        for u in pend:
            k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[6:10] /= np.linalg.norm(x[6:10])
        return x

    def update(self, t, state, flat_output):
        # command applied over the last step = issued delay_steps before it
        applied = self.cmd_hist[0] if self.cmd_hist else None
        omega_cmd_prev = np.sqrt(applied / self.k_eta_ctrl) if applied is not None else None
        if self.record is not None:
            self.record.append(({k: np.array(state[k], float) for k in ('x', 'v', 'q', 'w', 'rotor_speeds')},
                                omega_cmd_prev))
        if self.kf is not None:
            self.est_log.append(np.concatenate(self.kf.update(state, omega_cmd_prev)))
        if self.step % self.solve_every == 0:
            params = self._stage_params(t, state)
            q = state['q']
            x0 = np.concatenate([state['x'], state['v'], [q[3], q[0], q[1], q[2]], state['w']])
            if self.filt == 2 and getattr(self.kf, 'xf', None) is not None:
                # learned state filter: [p, v, w] (attitude stays measured)
                x0[0:6], x0[10:13] = self.kf.xf[0:6], self.kf.xf[6:9]
            elif self.filt and self.kf is not None:
                x0[3:6], x0[10:13] = self.kf.z[0:3], self.kf.z[3:6]
            t_ref = t
            if self.delay_steps:
                x0 = self._predict(x0, params[0])
                t_ref = t + self.delay_steps * self.sim_dt
            yref = np.zeros((self.N, 17))
            yref_e = None
            for k in range(self.N + 1):
                fl = self.trajectory.update(t_ref + k * self.T / self.N)
                pk = params[k]
                xr, ur = flat_to_state_ref(fl, pk[LAYOUT.slices['m']][0],
                                           pk[LAYOUT.slices['g']].sum())
                if np.dot(xr[6:10], x0[6:10]) < 0:
                    xr[6:10] *= -1
                if k < self.N:
                    yref[k] = np.concatenate([xr, ur])
                else:
                    yref_e = xr
            u, ts = self.mpc.solve(x0, yref, yref_e, params)
            if np.all(np.isfinite(u)):
                self.u = np.clip(u, 0, self.mpc.f_max)
            self.solve_times.append(ts)
        self.cmd_hist.append(self.u.copy())
        self.step += 1

        cmd_motor_speeds = np.sqrt(self.u / self.k_eta_ctrl)
        TM = self.f_to_TM @ self.u
        return {'cmd_motor_speeds': cmd_motor_speeds,
                'cmd_motor_thrusts': self.u.copy(),
                'cmd_thrust': TM[0],
                'cmd_moment': TM[1:],
                'cmd_q': np.array([0, 0, 0, 1.0]),
                'cmd_w': np.zeros(3)}
