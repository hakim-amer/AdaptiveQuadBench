"""
Hard evaluation regimes layered on top of any AdaptiveQuadBench experiment.

Each regime is applied identically to every controller: all randomness uses dedicated
per-trial RNG streams, so noise realisations, fault times and gusts are the same across
controllers and do not disturb the benchmark's own random stream (motor noise etc.).

    base        : benchmark as-is
    noise       : controller sees noisy pose/twist/rotor-speed telemetry (mocap + IMU level)
    noise_hi    : 3x noise
    aggr        : aggressive motion primitives (see AGGRESSIVE_TRAJ)
    rotor_fault : one rotor abruptly loses 40-70% effectiveness at a random time
    gust_front  : 3-6 m/s wind step (50 ms ramp) at a random time
    lat20/lat50 : 20 / 50 ms actuation latency
    lat_rand    : per-trial latency ~ U(0, 60 ms)
    combo       : noise + lat20 + aggr + rotor_fault
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

NOISE = dict(p=0.005, v=0.05, att=0.01, w=0.05, rpm=20.0)  # m, m/s, rad, rad/s, rad/s
AGGRESSIVE_TRAJ = dict(traj_pos_range=(-4, 4), traj_vel_range=(-6, 6), traj_acc_range=(-6, 6))

REGIMES = {
    'base': {},
    'noise': {'noise': 1.0},
    'noise_hi': {'noise': 3.0},
    'aggr': {'aggressive': True},
    'rotor_fault': {'rotor_fault': True},
    'gust_front': {'gust_front': True},
    'lat20': {'latency': 0.02},
    'lat50': {'latency': 0.05},
    'lat_rand': {'latency': 'random'},  # per-trial latency ~ U(0, 60 ms), unknown to the controller
    'combo': {'noise': 1.0, 'latency': 0.02, 'aggressive': True, 'rotor_fault': True},
}


def trajectory_kind(regime, trajectory):
    return 'aggressive' if REGIMES[regime].get('aggressive') and trajectory == 'random' else trajectory


def fault_spec(i):
    rng = np.random.default_rng(200000 + i)
    return dict(t=rng.uniform(1.5, 3.5), rotor=int(rng.integers(4)), eff=rng.uniform(0.3, 0.6))


def gust_spec(i):
    rng = np.random.default_rng(250000 + i)
    d = rng.normal(size=3)
    d[2] *= 0.3
    return dict(t=rng.uniform(1.5, 3.5), vec=rng.uniform(3, 6) * d / np.linalg.norm(d), ramp=0.05)


def apply_gust_front(wind_seq, i, dt, n_steps):
    g = gust_spec(i)
    seq = np.zeros((n_steps, 3)) if wind_seq is None else wind_seq.copy()
    t = np.arange(len(seq)) * dt
    seq += np.clip((t - g['t']) / g['ramp'], 0, 1)[:, None] * g['vec']
    return seq


def apply_rotor_fault(vehicle, i):
    """Abruptly scale one rotor's effectiveness mid-flight (Multirotor reads it live)."""
    f = fault_spec(i)
    step, clock = vehicle.step, {'t': 0.0, 'done': False}

    def faulty_step(state, control, t_step):
        if not clock['done'] and clock['t'] >= f['t'] - 1e-9:
            eff = np.array(vehicle.rotor_efficiency, dtype=float).copy()
            eff[f['rotor']] *= f['eff']
            vehicle.rotor_efficiency = eff
            clock['done'] = True
        out = step(state, control, t_step)
        clock['t'] += t_step
        return out
    vehicle.step = faulty_step


def wrap_sensor_noise(controller, i, scale):
    rng = np.random.default_rng(300000 + i)
    s = {k: v * scale for k, v in NOISE.items()}
    inner = controller.update

    def noisy_update(t, state, flat):
        st = dict(state)
        st['x'] = state['x'] + rng.normal(0, s['p'], 3)
        st['v'] = state['v'] + rng.normal(0, s['v'], 3)
        st['q'] = (Rotation.from_rotvec(rng.normal(0, s['att'], 3)) * Rotation.from_quat(state['q'])).as_quat()
        st['w'] = state['w'] + rng.normal(0, s['w'], 3)
        st['rotor_speeds'] = state['rotor_speeds'] + rng.normal(0, s['rpm'], len(state['rotor_speeds']))
        return inner(t, st, flat)
    controller.update = noisy_update


def latency_of(spec_latency, i):
    if spec_latency == 'random':
        return float(np.random.default_rng(350000 + i).uniform(0.0, 0.06))
    return spec_latency


def wrap_latency(controller, delay, dt):
    d = int(round(delay / dt))
    buf, inner = deque(), controller.update

    def delayed_update(t, state, flat):
        buf.append(inner(t, state, flat))
        if len(buf) > d + 1:
            buf.popleft()
        return buf[0]
    controller.update = delayed_update
