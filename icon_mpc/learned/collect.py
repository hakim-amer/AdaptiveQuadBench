"""
Collect estimator training data: KF-NMPC (behaviour policy) rollouts across experiments x regimes.

Trials are drawn from a different benchmark seed and trial-index range than evaluation
(eval: seed 42, trials 0..19), so vehicles, trajectories, disturbances and regime RNG streams
(keyed on the trial index) never overlap with the test set.

    python -m icon_mpc.learned.collect --out icon_mpc/results/data/v1 --trials 20 100
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from icon_mpc.headroom_study import REPO, _init_worker, run_task

BEHAVIOUR = ['nmpc+kf[q_F=10,r_v=0.05,r_w=0.05,q_tau=0.1,filt=1]',
             'nmpc+kf[q_F=10,r_v=0.15,r_w=0.15,q_tau=0.1,filt=1]',
             'nmpc+kf[q_F=1,r_v=0.15,r_w=0.15,q_tau=0.1,filt=1]',
             'nmpc+kf[q_F=30,r_v=0.001,r_w=0.001,q_tau=0.3,filt=1]']
NOISY = ('noise', 'noise_hi', 'combo')
SEQ_LEN = 500


def behaviour_for(regime, i, policy=None):
    if policy:
        return policy
    # the fast (low-R) KF destabilises the NMPC under sensor noise; skip it there
    pool = BEHAVIOUR[:3] if regime in NOISY else BEHAVIOUR
    return pool[i % len(pool)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', nargs='+', default=['no', 'wind', 'force', 'torque',
                                                          'payload', 'rotoreff', 'uncertainty'])
    ap.add_argument('--regimes', nargs='+', default=['base', 'noise', 'noise_hi', 'gust_front',
                                                      'rotor_fault', 'aggr'])
    ap.add_argument('--trials', nargs=2, type=int, default=[20, 100])
    ap.add_argument('--num_trials', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--policy', default=None, help='override behaviour controller (e.g. DAgger)')
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    _init_worker()
    from quad_param.quadrotor import quad_params
    from icon_mpc.nmpc import ParamNMPCSolver
    ParamNMPCSolver(quad_params['k_eta'] * quad_params['rotor_speed_max'] ** 2)
    os.chdir(REPO)

    tasks = [(e, behaviour_for(r, i, args.policy), i, args.num_trials, args.seed, 'random', r, 'collect')
             for r in args.regimes for e in args.experiments for i in range(*args.trials)]
    print(f'{len(tasks)} collection rollouts', flush=True)
    from concurrent.futures import as_completed
    from pebble import ProcessPool
    buf = {k: [] for k in ('feat', 'base', 'label', 'sbase', 'slabel')}
    meta, t0, n_fail = [], time.time(), 0
    with ProcessPool(max_workers=args.workers, initializer=_init_worker,
                     context=mp.get_context('spawn')) as pool:
        futs = {pool.schedule(run_task, args=(t,), timeout=60): t for t in tasks}
        for k, f in enumerate(as_completed(futs)):
            t = futs[f]
            try:
                m = f.result()
                d = m.pop('data')
            except Exception:
                d = None
            if d is None or len(d['label']) < SEQ_LEN:
                n_fail += 1
                continue
            for key in buf:
                buf[key].append(d[key][:SEQ_LEN])
            meta.append((t[6], t[0], t[2], t[1], m['rmse']))
            if (k + 1) % max(1, len(tasks) // 10) == 0:
                print(f'  {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s, {n_fail} failed)', flush=True)
    np.savez_compressed(os.path.join(args.out, 'data.npz'),
                        **{k: np.stack(v) for k, v in buf.items()},
                        regime=np.array([m[0] for m in meta]), experiment=np.array([m[1] for m in meta]),
                        trial=np.array([m[2] for m in meta]), policy=np.array([m[3] for m in meta]),
                        rmse=np.array([m[4] for m in meta]))
    print(f'saved {len(meta)} sequences ({n_fail} failed) to {args.out}')


if __name__ == '__main__':
    main()
