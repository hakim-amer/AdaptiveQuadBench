"""The NMPC prediction model must match RotorPy's Multirotor dynamics exactly."""
import copy
import numpy as np
import casadi as cs
from scipy.spatial.transform import Rotation
from quad_param.quadrotor import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from icon_mpc.nmpc import build_model, vehicle_params, LAYOUT


def _check(vehicle, k_eta_ctrl, rng, n=50):
    model = build_model('consistency')
    f = cs.Function('f', [model.x, model.u, model.p], [model.f_expl_expr])
    base = vehicle_params(vehicle, k_eta_ctrl)
    base[LAYOUT.slices['w0']] = 0.0  # exact rotor-speed mapping for the equivalence check
    worst = 0.0
    for _ in range(n):
        q = Rotation.random(random_state=rng.integers(1e9)).as_quat()
        v, w, wind = rng.normal(0, 2, 3), rng.normal(0, 2, 3), rng.normal(0, 2, 3)
        F, tau = rng.normal(0, 1, 3), rng.normal(0, 0.1, 3)
        u = rng.uniform(0.5, 6.0, 4)
        omega = np.sqrt(u / k_eta_ctrl)
        s = Multirotor._pack_state({'x': np.zeros(3), 'v': v, 'q': q, 'w': w,
                                    'wind': wind, 'rotor_speeds': omega})
        sd = vehicle._s_dot_fn(0, s, omega, F, tau)
        p = base.copy()
        p[LAYOUT.slices['F']], p[LAYOUT.slices['tau']], p[LAYOUT.slices['wind']] = F, tau, wind
        x = np.concatenate([np.zeros(3), v, [q[3], q[0], q[1], q[2]], w])
        xd = np.array(f(x, u, p)).ravel()
        sim = np.concatenate([sd[0:6], [sd[9], sd[6], sd[7], sd[8]], sd[10:13]])
        # rotor speed sqrt uses +1e-6 regularisation -> tiny relative error allowed
        worst = max(worst, np.max(np.abs(xd - sim) / (1 + np.abs(sim))))
    return worst


def test_nominal_vehicle():
    rng = np.random.default_rng(0)
    assert _check(Multirotor(quad_params), quad_params['k_eta'], rng) < 1e-4


def test_payload_rotoreff_and_mismatched_keta():
    rng = np.random.default_rng(1)
    qp = copy.deepcopy(quad_params)
    qp['rotor_efficiency'] = np.array([0.7, 1.2, 0.9, 1.1])
    veh = Multirotor(qp)
    veh.update_payload(0.4, np.array([0.03, -0.02, 0.01]))
    veh.attach_payload()
    assert _check(veh, 0.8 * quad_params['k_eta'], rng) < 1e-4
