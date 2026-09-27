"""
Parametric NMPC for AdaptiveQuadBench (acados, SQP-RTI).

The prediction model mirrors the RotorPy Multirotor wrench model and exposes
every physical / disturbance quantity as a *per-stage runtime parameter*, so
that oracles (and later the learned in-context estimator) can inject
time-varying model information over the horizon without recompiling.

State  x = [p(3), v(3), q_wxyz(4), w(3)]
Input  u = commanded nominal rotor thrusts f_i = k_eta_ctrl * omega_cmd_i^2  [N]

Per-stage parameter vector (see ParamLayout):
    m, J(6: xx,yy,zz,xy,xz,yz), r_i (4x3 rotor positions w.r.t. CoM),
    g_i (4, thrust gain = true_thrust / f_i), kappa_i (4, signed yaw moment per f_i),
    k_d, k_z (rotor drag), F_ext (3, world), tau_ext (3, body), wind (3, world),
    keta (controller's k_eta, maps commanded thrust to rotor speed for rotor drag),
    w0 (rotor-speed floor in the drag model: omega = sqrt(f/keta + w0^2); keeps d omega/d f
        bounded near zero thrust, otherwise the optimizer exploits a spurious drag lever)
"""

import fcntl
import glob
import os
import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

GRAV = 9.81
DRAG_OMEGA_FLOOR = 50.0  # rad/s
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')


class ParamLayout:
    names = [('m', 1), ('J', 6), ('r', 12), ('g', 4), ('kappa', 4),
             ('kd', 1), ('kz', 1), ('F', 3), ('tau', 3), ('wind', 3), ('keta', 1), ('w0', 1)]

    def __init__(self):
        self.slices, i = {}, 0
        for n, d in self.names:
            self.slices[n] = slice(i, i + d)
            i += d
        self.dim = i

    def pack(self, **kw):
        p = np.zeros(self.dim)
        for n, v in kw.items():
            p[self.slices[n]] = np.asarray(v, dtype=float).reshape(-1)
        return p


LAYOUT = ParamLayout()


def nominal_params(quad_params, k_eta_ctrl=None, aero=False):
    """Parameter vector implied by a RotorPy quad_params dict (no disturbances)."""
    k_eta_ctrl = quad_params['k_eta'] if k_eta_ctrl is None else k_eta_ctrl
    com = np.asarray(quad_params.get('com', np.zeros(3)), dtype=float)
    r = np.array([quad_params['rotor_pos'][k] - com for k in quad_params['rotor_pos']])
    eff = np.asarray(quad_params.get('rotor_efficiency', np.ones(4)), dtype=float)
    dirs = np.asarray(quad_params['rotor_directions'], dtype=float)
    J = [quad_params['Ixx'], quad_params['Iyy'], quad_params['Izz'],
         quad_params['Ixy'], quad_params['Ixz'], quad_params['Iyz']]
    return LAYOUT.pack(
        m=quad_params['mass'], J=J, r=r,
        g=eff * quad_params['k_eta'] / k_eta_ctrl,
        kappa=dirs * eff * quad_params['k_m'] / k_eta_ctrl,
        kd=quad_params['k_d'] if aero else 0.0,
        kz=quad_params['k_z'] if aero else 0.0, keta=k_eta_ctrl, w0=DRAG_OMEGA_FLOOR)


def vehicle_params(vehicle, k_eta_ctrl, aero=True):
    """Ground-truth parameter vector read from a live rotorpy Multirotor object."""
    I = vehicle.inertia
    J = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    eff = np.asarray(vehicle.rotor_efficiency, dtype=float)
    return LAYOUT.pack(
        m=vehicle.mass, J=J, r=vehicle.rotor_geometry.copy(),
        g=eff * vehicle.k_eta / k_eta_ctrl,
        kappa=vehicle.rotor_dir * eff * vehicle.k_m / k_eta_ctrl,
        kd=vehicle.k_d if aero else 0.0, kz=vehicle.k_z if aero else 0.0, keta=k_eta_ctrl,
        w0=DRAG_OMEGA_FLOOR)


def _quat_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return cs.vertcat(
        cs.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        cs.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        cs.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def build_model(name):
    x = cs.SX.sym('x', 13)
    u = cs.SX.sym('u', 4)
    p = cs.SX.sym('p', LAYOUT.dim)
    P = {n: p[s] for n, s in LAYOUT.slices.items()}

    vel, q, w = x[3:6], x[6:10], x[10:13]
    R = _quat_rot(q)
    Jv = P['J']
    J = cs.vertcat(cs.horzcat(Jv[0], Jv[3], Jv[4]),
                   cs.horzcat(Jv[3], Jv[1], Jv[5]),
                   cs.horzcat(Jv[4], Jv[5], Jv[2]))
    v_air_b = R.T @ (vel - P['wind'])
    Dm = cs.diag(cs.vertcat(P['kd'], P['kd'], P['kz']))

    F_b = cs.SX.zeros(3)
    M_b = cs.SX.zeros(3)
    for i in range(4):
        r_i = P['r'][3 * i:3 * i + 3]
        omega_i = cs.sqrt(cs.fmax(u[i], 0.0) / P['keta'] + P['w0'] ** 2 + 1e-6)
        v_loc = v_air_b + cs.cross(w, r_i)
        H_i = -omega_i * (Dm @ v_loc)
        F_i = cs.vertcat(0, 0, P['g'][i] * u[i]) + H_i
        F_b += F_i
        M_b += cs.cross(r_i, F_i) + cs.vertcat(0, 0, P['kappa'][i] * u[i])

    Om = cs.vertcat(cs.horzcat(0, -w[0], -w[1], -w[2]),
                    cs.horzcat(w[0], 0, w[2], -w[1]),
                    cs.horzcat(w[1], -w[2], 0, w[0]),
                    cs.horzcat(w[2], w[1], -w[0], 0))
    q_dot = 0.5 * Om @ q
    v_dot = (R @ F_b + P['F']) / P['m'] - cs.vertcat(0, 0, GRAV)
    w_dot = cs.solve(J, M_b + P['tau'] - cs.cross(w, J @ w))
    f_expl = cs.vertcat(vel, v_dot, q_dot, w_dot)

    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = x, u, p
    model.xdot = cs.SX.sym('xdot', 13)
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl
    return model


def _default_params():
    p = np.zeros(LAYOUT.dim)
    p[LAYOUT.slices['m']] = 1.0
    p[LAYOUT.slices['J']] = [1e-2, 1e-2, 1e-2, 0, 0, 0]
    p[LAYOUT.slices['g']] = 1.0
    p[LAYOUT.slices['keta']] = 1e-5
    return p


_SOLVER_CACHE = {}


def get_param_nmpc(f_max, t_horizon=0.5, n_nodes=10):
    """Per-process cached solver, reset to a cold start so rollouts stay independent."""
    key = (t_horizon, n_nodes)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = ParamNMPCSolver(f_max, t_horizon, n_nodes)
    solver = _SOLVER_CACHE[key]
    solver.solver.reset()
    solver.n_solves = solver.n_fail = 0
    solver.set_input_bounds(f_max)
    return solver


class ParamNMPCSolver:
    """Thin wrapper around a compiled acados OCP (compile once, load many)."""

    Q = np.array([10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    R = np.array([0.01, 0.01, 0.01, 0.01])

    def __init__(self, f_max, t_horizon=0.5, n_nodes=10, name='icon_param_nmpc'):
        self.N, self.T = n_nodes, t_horizon
        self.n_solves = self.n_fail = 0
        os.makedirs(BUILD_DIR, exist_ok=True)
        code_dir = os.path.join(BUILD_DIR, f'{name}_N{n_nodes}')
        json_file = os.path.join(BUILD_DIR, f'{name}_N{n_nodes}.json')

        ocp = AcadosOcp()
        ocp.model = build_model(name)
        ocp.code_export_directory = code_dir
        ocp.solver_options.N_horizon = n_nodes
        ocp.solver_options.tf = t_horizon
        ocp.parameter_values = _default_params()
        ny = 17
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.diag(np.concatenate([self.Q, self.R]))
        ocp.cost.W_e = np.diag(self.Q)
        ocp.cost.Vx = np.zeros((ny, 13))
        ocp.cost.Vx[:13, :13] = np.eye(13)
        ocp.cost.Vu = np.zeros((ny, 4))
        ocp.cost.Vu[13:, :] = np.eye(4)
        ocp.cost.Vx_e = np.eye(13)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(13)
        ocp.constraints.x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.])
        ocp.constraints.lbu = np.zeros(4)
        ocp.constraints.ubu = np.full(4, 10.0)
        ocp.constraints.idxbu = np.arange(4)
        so = ocp.solver_options
        so.qp_solver = 'FULL_CONDENSING_HPIPM'
        so.hessian_approx = 'GAUSS_NEWTON'
        so.integrator_type = 'ERK'
        so.nlp_solver_type = 'SQP_RTI'
        so.print_level = 0
        cwd = os.getcwd()
        try:
            # compile once (file lock guards concurrent workers), then only load
            with open(os.path.join(BUILD_DIR, f'{name}_N{n_nodes}.lock'), 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                build = not glob.glob(os.path.join(code_dir, 'libacados_ocp_solver_*.so'))
                self.solver = AcadosOcpSolver(ocp, json_file=json_file, build=build,
                                              generate=build, verbose=False)
        finally:
            os.chdir(cwd)
        self.set_input_bounds(f_max)

    def set_input_bounds(self, f_max):
        self.f_max = f_max
        ub = np.full(4, f_max)
        for k in range(self.N):
            self.solver.constraints_set(k, 'ubu', ub)

    def solve(self, x0, yref, yref_e, params):
        """yref: (N, 17); yref_e: (13,); params: (N+1, np) per-stage parameters."""
        s = self.solver
        s.set(0, 'lbx', x0)
        s.set(0, 'ubx', x0)
        for k in range(self.N):
            s.set(k, 'yref', yref[k])
            s.set(k, 'p', params[k])
        s.set(self.N, 'yref', yref_e)
        s.set(self.N, 'p', params[self.N])
        status = s.solve()
        u = s.get(0, 'u')
        self.n_solves += 1
        # SQP-RTI can report success with an exploded predicted trajectory (QP hit its iteration
        # cap); warm-starting from such iterates makes the next HPIPM call hang forever.
        X = np.stack([s.get(k, 'x') for k in range(self.N + 1)])
        bad_traj = (not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3
                    or np.abs(np.linalg.norm(X[:, 6:10], axis=1) - 1).max() > 0.5)
        if status not in (0, 2) or not np.all(np.isfinite(u)) or bad_traj:
            self.n_fail += 1
            self.cold_start(x0)
            u = np.full(4, np.nan)
        return u, s.get_stats('time_tot')

    def cold_start(self, x0):
        """Reset solver memory and initialise iterates at x0 / hover-ish thrust (not zeros)."""
        s = self.solver
        s.reset()
        for k in range(self.N + 1):
            s.set(k, 'x', x0)
        for k in range(self.N):
            s.set(k, 'u', np.full(4, 0.25 * self.f_max))
