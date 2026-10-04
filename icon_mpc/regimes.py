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
    mix         : per-trial random combination (noise scale, latency, aggr, fault, gust); training only
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
    'mix': {'mix': True},
    'mix2': {'mix': 2},  # mix + non-Gaussian noise / outliers / impulses / flicker faults; training only
    # non-Gaussian sensing / unmodelled events (stress tests; not used for training)
    'heavy': {'noise': 1.0, 'heavy': 2.5},           # Student-t (nu=2.5) instead of Gaussian noise
    'outlier': {'noise': 1.0, 'outlier': 0.02},      # 2 %/step per-channel glitches of 20 sigma
    'dropout': {'noise': 1.0, 'dropout': True},      # mocap/IMU freezes (stale packets) of 50-200 ms
    'impulse': {'impulse': True},                    # collision-like velocity / body-rate kicks
    'flicker': {'flicker': True},                    # intermittent rotor fault (toggles on/off)
    'stress': {'noise': 1.0, 'outlier': 0.02, 'dropout': True, 'impulse': True, 'flicker': True,
               'aggressive': True},
}


def spec_for(regime, i):
    """Concrete regime spec for trial i ('mix' draws a random combination per trial)."""
    spec = REGIMES[regime]
    if not spec.get('mix'):
        return spec
    rng = np.random.default_rng(400000 + i)
    out = {}
    if rng.random() < 0.75:
        out['noise'] = float(rng.uniform(0.3, 3.0))
    if rng.random() < 0.5:
        out['latency'] = float(rng.uniform(0.0, 0.04))
    if rng.random() < 0.5:
        out['aggressive'] = True
    if rng.random() < 0.5:
        out['rotor_fault'] = True
    if rng.random() < 0.3:
        out['gust_front'] = True
    if spec['mix'] == 2:
        r2 = np.random.default_rng(480000 + i)
        if 'noise' in out and r2.random() < 0.3:
            out['heavy'] = 2.5
        if 'noise' in out and r2.random() < 0.3:
            out['outlier'] = 0.02
        if r2.random() < 0.3:
            out['impulse'] = True
        if r2.random() < 0.3:
            out.pop('rotor_fault', None)
            out['flicker'] = True
    return out


def trajectory_kind(spec, trajectory):
    return 'aggressive' if spec.get('aggressive') and trajectory == 'random' else trajectory


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


def wrap_sensor_noise(controller, i, scale, heavy=None, outlier=0.0, dropout=False):
    """Sensor corruption. heavy: Student-t dof (unit-variance-scaled) instead of Gaussian;
    outlier: per-step, per-channel-group probability of a 20-sigma glitch; dropout: bursts in
    which the controller keeps receiving the last packet (stale, bit-identical measurements)."""
    rng = np.random.default_rng(300000 + i)
    s = {k: v * scale for k, v in NOISE.items()}
    inner = controller.update
    held = {'left': 0, 'st': None}

    def draw(sd, n):
        if heavy:
            e = rng.standard_t(heavy, n) * np.sqrt((heavy - 2) / heavy)
        else:
            e = rng.normal(0, 1, n)
        if outlier and rng.random() < outlier:
            e = e + rng.normal(0, 20, n)
        return sd * e

    def noisy_update(t, state, flat):
        if dropout:
            if held['left'] > 0 and held['st'] is not None:
                held['left'] -= 1
                return inner(t, held['st'], flat)
            if t > 0.5 and rng.random() < 0.01:
                held['left'] = int(rng.integers(5, 21))
        st = dict(state)
        st['x'] = state['x'] + draw(s['p'], 3)
        st['v'] = state['v'] + draw(s['v'], 3)
        st['q'] = (Rotation.from_rotvec(draw(s['att'], 3)) * Rotation.from_quat(state['q'])).as_quat()
        st['w'] = state['w'] + draw(s['w'], 3)
        st['rotor_speeds'] = state['rotor_speeds'] + draw(s['rpm'], len(state['rotor_speeds']))
        held['st'] = st
        return inner(t, st, flat)
    controller.update = noisy_update


def apply_impulses(vehicle, i):
    """Collision-like kicks: 3 instantaneous velocity (0.5-1.5 m/s) and body-rate (2-5 rad/s) jumps."""
    rng = np.random.default_rng(450000 + i)
    kicks = sorted(rng.uniform(1.0, 4.5, 3))
    dv = [rng.uniform(0.5, 1.5) * (lambda d: d / np.linalg.norm(d))(rng.normal(size=3)) for _ in kicks]
    dw = [rng.uniform(2, 5) * (lambda d: d / np.linalg.norm(d))(rng.normal(size=3)) for _ in kicks]
    step, clock = vehicle.step, {'t': 0.0, 'k': 0}

    def kicked_step(state, control, t_step):
        out = step(state, control, t_step)
        clock['t'] += t_step
        k = clock['k']
        if k < len(kicks) and clock['t'] >= kicks[k]:
            out = dict(out)
            out['v'] = out['v'] + dv[k]
            out['w'] = out['w'] + dw[k]
            clock['k'] += 1
        return out
    vehicle.step = kicked_step
    vehicle.kick_times = kicks


def apply_flicker_fault(vehicle, i):
    """Intermittent rotor fault: one rotor toggles between healthy and 30-60 % effectiveness
    every 0.2-0.8 s after a random onset (loose connector / ESC brown-out)."""
    f = fault_spec(i)
    rng = np.random.default_rng(470000 + i)
    toggles = np.cumsum(np.concatenate([[f['t'] - 0.5], rng.uniform(0.2, 0.8, 30)]))
    step, clock = vehicle.step, {'t': 0.0, 'k': 0}
    healthy = np.array(vehicle.rotor_efficiency, dtype=float).copy()

    def flick_step(state, control, t_step):
        while clock['k'] < len(toggles) and clock['t'] >= toggles[clock['k']]:
            clock['k'] += 1
            eff = healthy.copy()
            if clock['k'] % 2 == 1:
                eff[f['rotor']] *= f['eff']
            vehicle.rotor_efficiency = eff
        out = step(state, control, t_step)
        clock['t'] += t_step
        return out
    vehicle.step = flick_step


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
