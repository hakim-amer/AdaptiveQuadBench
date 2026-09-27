"""
Oracle headroom study for ICON-MPC.

Runs benchmark baselines and privileged-oracle NMPC variants on *identical*
AdaptiveQuadBench trials (same trajectories, vehicles, disturbances, wind
realisation and motor-noise seed) and reports how much tracking error each
piece of model knowledge removes.

    python -m icon_mpc.headroom_study --experiments no wind payload \
        --controllers indi-a l1mpc nmpc nmpc+aero nmpc+params nmpc+dist nmpc+future
"""

import argparse
import copy
import csv
import multiprocessing as mp
import os
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_FINAL, SIM_DT = 5.0, 0.01

ORACLES = {
    'nmpc20': ('nominal', 5),
    'nmpc': ('nominal', 1),
    'nmpc+aero': ('aero', 1),
    'nmpc+params': ('params', 1),
    'nmpc+dist': ('dist', 1),
    'nmpc+future': ('future', 1),
    # estimator-delay sweep: disturbance/parameter truth delayed by the given lag
    'nmpc+dist@0ms': ('dist', 1, 0.0),
    'nmpc+dist@30ms': ('dist', 1, 0.03),
    'nmpc+dist@100ms': ('dist', 1, 0.1),
    'nmpc+dist@300ms': ('dist', 1, 0.3),
}
BASELINES = ['geo', 'geo-a', 'l1geo', 'indi-a', 'mpc', 'l1mpc', 'xadap']


def _patch_baseline_acados():
    """Baseline MPC bakes vehicle params into its model, and acados' auto-generated OCP
    name/hash is not deterministic across calls. Give each OCP a deterministic name and
    code dir keyed on the baked-in params; build once per key under a per-key file lock."""
    import fcntl
    import hashlib
    import controller.quadrotor_traopt as traopt
    orig = traopt.AcadosOcpSolver
    if getattr(orig, '_icon_patched', False):
        return
    build_root = os.path.join(REPO, 'icon_mpc', 'build', 'baseline_mpc')
    os.makedirs(build_root, exist_ok=True)
    keys = ['mass', 'Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz', 'k_m', 'k_eta',
            'rotor_speed_min', 'rotor_speed_max']

    def solver(ocp, json_file, verbose=False, generate=True, build=True):
        qp = sys._getframe(1).f_locals['quad_params']
        blob = repr([float(qp[k]) for k in keys] +
                    [np.round(qp['rotor_pos'][r], 12).tolist() for r in qp['rotor_pos']] +
                    [ocp.solver_options.tf, ocp.solver_options.N_horizon])
        key = 'qmpc_' + hashlib.md5(blob.encode()).hexdigest()[:10]
        ocp.name = key
        ocp.code_export_directory = os.path.join(build_root, key)
        jf = os.path.join(build_root, f'{key}.json')
        with open(os.path.join(build_root, f'{key}.lock'), 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            done = os.path.exists(os.path.join(build_root, f'{key}.ok'))
            s = orig(ocp, json_file=jf, verbose=False, generate=not done, build=not done)
            if not done:
                open(os.path.join(build_root, f'{key}.ok'), 'w').close()
            return s

    solver._icon_patched = True
    traopt.AcadosOcpSolver = solver


def _init_worker():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)
    _patch_baseline_acados()


class RecordedWind:
    """Replays a pre-sampled wind sequence so the oracle can see the future."""

    def __init__(self, seq, dt=SIM_DT):
        self.seq, self.dt = seq, dt

    def update(self, t, position):
        i = int(round(t / self.dt))
        return self.seq[min(i, len(self.seq) - 1)].copy()


@lru_cache(maxsize=None)
def get_components(experiment, num_trials, seed, trajectory):
    from quad_param.quadrotor import quad_params
    from config.randomization_config import RandomizationConfig
    from rotorpy.vehicles.multirotor import Multirotor

    cfg = RandomizationConfig.from_experiment_type(experiment, num_trials, quad_params, seed,
                                                   trajectory_type=trajectory)
    comps = {
        'trajectories': cfg.create_trajectories(),
        'wind_profiles': cfg.create_wind_profiles(),
    }
    comps['ext_force'], comps['ext_torque'] = cfg.create_ext_force_and_torque()
    masses, positions = cfg.create_payload_disturbance()
    comps['toggle_times'] = cfg.create_disturbance_toggle_times()
    vehicles = [Multirotor(p) for p in cfg.create_vehicle_params(quad_params)]
    if masses is not None:
        for v, m, pos in zip(vehicles, masses, positions):
            v.update_payload(m, pos)
    comps['vehicles'] = vehicles
    comps['controller_params'] = cfg.create_controller_params(quad_params)

    horizon_steps = int((T_FINAL + 1.0) / SIM_DT) + 2
    winds = []
    for i, wp in enumerate(comps['wind_profiles']):
        if wp is None:
            winds.append(None)
            continue
        np.random.seed(100000 + i)
        wp = copy.deepcopy(wp)
        winds.append(np.array([wp.update(k * SIM_DT, None) for k in range(horizon_steps)]))
    comps['wind_seqs'] = winds
    return comps


ROLLOUT_TIMEOUT_S = 60
TASK_TIMEOUT_S = 120


def compute_metrics(res):
    from scipy.spatial.transform import Rotation
    t = res['time']
    x, xd = res['state']['x'], res['flat']['x']
    err = np.linalg.norm(x - xd, axis=1)
    yaw = Rotation.from_quat(res['state']['q']).as_euler('xyz')[:, 2]
    dyaw = yaw - res['flat']['yaw']
    wrapped = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    late = t > 1.0
    u = res['control']['cmd_motor_speeds']
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'rmse_after1s': float(np.sqrt(np.mean(err[late] ** 2))),
        'max_err': float(err.max()),
        'heading_deg': float(np.rad2deg(np.abs(dyaw).mean())),
        'heading_wrapped_deg': float(np.rad2deg(np.abs(wrapped).mean())),
        'cmd_rate': float(np.mean(np.abs(np.diff(u, axis=0))) / SIM_DT),
    }


def run_task(task):
    experiment, ctrl_name, i, num_trials, seed, trajectory = task
    from rotorpy.environments import Environment
    from run_eval import switch_controller
    from icon_mpc.oracle_controller import OracleNMPC, Privileged

    c = get_components(experiment, num_trials, seed, trajectory)
    vehicle = copy.deepcopy(c['vehicles'][i])
    traj = c['trajectories'][i]
    wind_seq = c['wind_seqs'][i]
    ext_f = c['ext_force'][i] if c['ext_force'] is not None else None
    ext_t = c['ext_torque'][i] if c['ext_torque'] is not None else None
    toggles = c['toggle_times'][i] if c['toggle_times'] is not None else None
    cparams = c['controller_params'][i]

    if ctrl_name in ORACLES:
        level, every, *lag = ORACLES[ctrl_name]
        priv = Privileged(vehicle, wind_seq, ext_f, ext_t, toggles, SIM_DT)
        controller = OracleNMPC(cparams, level=level, privileged=priv, solve_every=every,
                                dist_lag=lag[0] if lag else None)
    else:
        controller = switch_controller(ctrl_name, cparams)
    os.chdir(REPO)
    controller.update_trajectory(traj)

    env = Environment(vehicle=vehicle, controller=controller,
                      wind_profile=RecordedWind(wind_seq) if wind_seq is not None else None,
                      trajectory=traj, sim_rate=int(1 / SIM_DT),
                      ext_force=ext_f, ext_torque=ext_t, disturbance_toggle_times=toggles)
    env.vehicle.initial_state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0, 0, 0, 1.]),
                                 'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.zeros(4)}
    np.random.seed(i)
    t0 = time.time()
    import faulthandler
    faulthandler.dump_traceback_later(TASK_TIMEOUT_S - 20, exit=False)
    _update = controller.update

    def guarded_update(*a, **kw):
        # Diverged rollouts make RK45 extremely stiff; abort them and count as failure.
        if time.time() - t0 > ROLLOUT_TIMEOUT_S:
            raise TimeoutError(f'rollout exceeded {ROLLOUT_TIMEOUT_S}s')
        return _update(*a, **kw)
    controller.update = guarded_update
    _sdot = vehicle._s_dot_fn

    def guarded_sdot(*a, **kw):
        # RK45 can stall inside a single step when the state blows up.
        if time.time() - t0 > ROLLOUT_TIMEOUT_S:
            raise TimeoutError(f'rollout exceeded {ROLLOUT_TIMEOUT_S}s')
        return _sdot(*a, **kw)
    vehicle._s_dot_fn = guarded_sdot
    try:
        res = env.run(t_final=T_FINAL, use_mocap=False, terminate=False, plot=False,
                      animate_bool=False, verbose=False)
        m = compute_metrics(res)
    except Exception as e:  # diverged solver etc. -> count as failure
        m = {'rmse': np.inf, 'rmse_after1s': np.inf, 'max_err': np.inf, 'heading_deg': np.nan,
             'heading_wrapped_deg': np.nan, 'cmd_rate': np.nan, 'error': repr(e)[:200]}
    faulthandler.cancel_dump_traceback_later()
    m.update(experiment=experiment, controller=ctrl_name, trial=i, wall_s=time.time() - t0,
             solve_fail=getattr(getattr(controller, 'mpc', None), 'n_fail', np.nan),
             solve_ms=float(np.mean(controller.solve_times) * 1e3) if getattr(controller, 'solve_times', None) else np.nan)
    return m


def prebuild_task(task):
    """Compile the baseline-MPC solver for one trial's controller params (unique hash)."""
    experiment, i, num_trials, seed, trajectory = task
    from controller.quadrotor_control_mpc import ModelPredictiveControl
    c = get_components(experiment, num_trials, seed, trajectory)
    ModelPredictiveControl(c['controller_params'][i]).update_trajectory(c['trajectories'][i])
    os.chdir(REPO)
    return i


def summarize(df):
    rows = []
    for (exp, ctrl), g in df.groupby(['experiment', 'controller'], sort=False):
        ok = g['rmse'] < 5
        s = g[ok]
        rows.append({'experiment': exp, 'controller': ctrl, 'n': len(g),
                     'success_%': 100 * ok.mean(),
                     'rmse': f"{s.rmse.mean():.4f} ± {s.rmse.std():.4f}",
                     'rmse>1s': f"{s.rmse_after1s.mean():.4f} ± {s.rmse_after1s.std():.4f}",
                     'heading_deg': f"{s.heading_wrapped_deg.mean():.3f}",
                     'solve_ms': f"{g.solve_ms.mean():.2f}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', nargs='+', default=['no', 'wind', 'force', 'torque',
                                                          'payload', 'rotoreff', 'uncertainty'])
    ap.add_argument('--controllers', nargs='+',
                    default=['indi-a', 'l1mpc', 'mpc'] + list(ORACLES))
    ap.add_argument('--num_trials', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trajectory', default='random')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(REPO, 'icon_mpc', 'results', 'headroom.csv'))
    args = ap.parse_args()
    _init_worker()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Compile acados solvers once in the parent so workers only load them.
    from quad_param.quadrotor import quad_params
    from icon_mpc.nmpc import ParamNMPCSolver
    ParamNMPCSolver(quad_params['k_eta'] * quad_params['rotor_speed_max'] ** 2)
    if any(c in ('mpc', 'l1mpc', 'xadap') for c in args.controllers):
        run_task(('no', 'mpc', 0, 1, args.seed, args.trajectory))
        if 'uncertainty' in args.experiments:
            pre = [('uncertainty', i, args.num_trials, args.seed, args.trajectory)
                   for i in range(args.num_trials)]
            print(f'prebuilding {len(pre)} baseline-MPC solvers (perturbed controller params)')
            with mp.get_context('spawn').Pool(args.workers, initializer=_init_worker) as pool:
                list(pool.imap_unordered(prebuild_task, pre))

    tasks = [(e, c, i, args.num_trials, args.seed, args.trajectory)
             for e in args.experiments for c in args.controllers for i in range(args.num_trials)]
    print(f'{len(tasks)} rollouts on {args.workers} workers')
    rows, t0 = [], time.time()
    # pebble kills workers stuck in C code (acados/scipy) that Python-level guards cannot interrupt
    from concurrent.futures import as_completed
    from pebble import ProcessPool
    with ProcessPool(max_workers=args.workers, initializer=_init_worker,
                     context=mp.get_context('spawn')) as pool:
        futs = {pool.schedule(run_task, args=(t,), timeout=TASK_TIMEOUT_S): t for t in tasks}
        for k, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:
                exp, ctrl, i = futs[f][:3]
                r = {'rmse': np.inf, 'rmse_after1s': np.inf, 'max_err': np.inf,
                     'heading_deg': np.nan, 'heading_wrapped_deg': np.nan, 'cmd_rate': np.nan,
                     'error': f'worker: {e!r}'[:200], 'experiment': exp, 'controller': ctrl,
                     'trial': i, 'wall_s': np.nan, 'solve_ms': np.nan}
                print(f'  FAILED {exp}/{ctrl}/{i}: {e!r}', flush=True)
            rows.append(r)
            if (k + 1) % max(1, len(tasks) // 20) == 0:
                print(f'  {k + 1}/{len(tasks)}  ({time.time() - t0:.0f}s)', flush=True)
                pd.DataFrame(rows).to_csv(args.out.replace('.csv', '_partial.csv'), index=False)
    df = pd.DataFrame(rows).sort_values(['experiment', 'controller', 'trial'])
    df.to_csv(args.out, index=False)
    summ = summarize(df)
    summ.to_csv(args.out.replace('.csv', '_summary.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
