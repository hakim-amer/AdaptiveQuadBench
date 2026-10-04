"""3D animations of benchmark rollouts: several controllers flying the same trial side by side.

  python -m icon_mpc.visualize --experiment wind --regime obsx_stress --trial 3 \
      --controllers mpc=Stock MPC "nmpc+learned[...]"=ICON-MPC --out icon_mpc/results/media/obs.mp4

Rollouts are (re)simulated with headroom_study.run_task and cached as .npz in --cache.
"""
import argparse
import os

import numpy as np

ARM = 0.16  # drawn arm length (m); exaggerated vs the real airframe for visibility
COLORS = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']


def _tag(experiment, regime, trial, ctrl):
    return f'{experiment}_{regime}_{trial}_{ctrl}'.replace('/', '_')


def simulate(experiment, regime, trial, controllers, cache, num_trials=20, seed=42, workers=8):
    cache = os.path.abspath(cache)  # stock controllers chdir during construction
    os.environ['ICON_SAVE_TRAJ'] = cache
    todo = [c for c in controllers if not os.path.exists(os.path.join(cache, _tag(experiment, regime, trial, c) + '.npz'))]
    if todo:
        import multiprocessing as mp
        from icon_mpc.headroom_study import run_task, _init_worker
        tasks = [(experiment, c, trial, num_trials, seed, 'random', regime) for c in todo]
        with mp.get_context('spawn').Pool(min(workers, len(tasks)), initializer=_init_worker) as pool:
            for m in pool.imap_unordered(run_task, tasks):
                print(f"  {m['controller'][:60]:60s} rmse={m['rmse']:.4f}", m.get('error', ''))
    out = []
    for c in controllers:
        f = os.path.join(cache, _tag(experiment, regime, trial, c) + '.npz')
        out.append(dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None)
    return out


def _clearance(x, obs):
    if len(obs) == 0:
        return np.full(len(x), np.inf)
    return np.min(np.linalg.norm(x[:, None, :2] - obs[None, :, :2], axis=-1) - obs[None, :, 2], axis=1)


def animate(runs, labels, out, title='', speed=0.5, fps=30, tail_s=1.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.spatial.transform import Rotation
    import imageio_ffmpeg
    from icon_mpc.regimes import R_BODY
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

    ok = [(r, l, COLORS[k % len(COLORS)] if k < len(runs) - 1 else '#2ca02c')
          for k, (r, l) in enumerate(zip(runs, labels)) if r is not None]
    ref = ok[-1][0]
    t, xd, obs = ref['time'], ref['xd'], ref['obstacles']
    obs = obs[obs[:, 2] > 0] if len(obs) else obs
    dt = t[1] - t[0]
    n_frames = int(t[-1] / speed * fps)
    idx = np.minimum((np.arange(n_frames) * speed / fps / dt).astype(int), len(t) - 1)
    tail = int(tail_s / dt)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(title, fontsize=15, weight='bold')
    ax3 = fig.add_axes([0.0, 0.03, 0.6, 0.9], projection='3d')
    axt = fig.add_axes([0.64, 0.42, 0.33, 0.48])
    axe = fig.add_axes([0.64, 0.07, 0.33, 0.27])

    allx = np.concatenate([xd] + [r['x'] for r, _, _ in ok])
    allx = allx[np.all(np.isfinite(allx), axis=1) & (np.linalg.norm(allx - xd.mean(0), axis=1) < 6)]
    lo, hi = allx.min(0) - 0.3, allx.max(0) + 0.3
    c, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax3.set_xlim(c[0] - half, c[0] + half); ax3.set_ylim(c[1] - half, c[1] + half)
    ax3.set_zlim(max(lo[2], c[2] - half), max(lo[2], c[2] - half) + 2 * half)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.view_init(elev=28, azim=-60)
    ax3.plot(*xd.T, '--', color='k', lw=1.2, alpha=0.6, label='reference')
    zr = np.linspace(ax3.get_zlim()[0], ax3.get_zlim()[1], 2)
    th = np.linspace(0, 2 * np.pi, 40)
    for cx, cy, r in obs:
        rp = r - R_BODY  # physical obstacle radius (r is inflated by the vehicle footprint)
        TH, Z = np.meshgrid(th, zr)
        ax3.plot_surface(cx + rp * np.cos(TH), cy + rp * np.sin(TH), Z, color='0.45', alpha=0.35, linewidth=0)
        axt.add_patch(plt.Circle((cx, cy), rp, color='0.45', alpha=0.6))
        axt.add_patch(plt.Circle((cx, cy), r, fill=False, ls=':', color='0.3'))
    axt.plot(xd[:, 0], xd[:, 1], '--', color='k', lw=1, alpha=0.6)
    axt.set_aspect('equal'); axt.set_title('top view (dotted: collision boundary)', fontsize=10)
    axt.set_xlim(c[0] - half, c[0] + half); axt.set_ylim(c[1] - half, c[1] + half)
    axe.set_xlim(0, t[-1]); axe.set_xlabel('t [s]'); axe.set_ylabel('tracking error [m]')
    axe.set_yscale('log'); axe.grid(alpha=0.3)

    rot_pts = ARM * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]) @ \
        Rotation.from_euler('z', 45, degrees=True).as_matrix().T
    ring = 0.35 * ARM * np.stack([np.cos(th), np.sin(th), 0 * th], 1)
    arts = []
    for r, lab, col in ok:
        x = r['x']
        err = np.linalg.norm(x - xd[:len(x)], axis=1)
        cl = _clearance(x, obs)
        rm = float(r['rmse'])
        n_coll_frames = int(np.sum(cl < 0))
        lab2 = f"{lab}  (RMSE {rm * 100:.1f} cm" + (f", {n_coll_frames * dt:.2f} s in collision)" if len(obs) else ')')
        axe.plot(t[:len(err)], np.maximum(err, 1e-4), color=col, lw=1, alpha=0.25)
        a = dict(x=x, R=Rotation.from_quat(r['q']).as_matrix(), err=err, cl=cl, col=col,
                 trail=ax3.plot([], [], [], color=col, lw=2.5, label=lab2)[0],
                 path=ax3.plot([], [], [], color=col, lw=1, alpha=0.45)[0],
                 arms=[ax3.plot([], [], [], color=col, lw=3)[0] for _ in range(2)],
                 rot=[ax3.plot([], [], [], color=col, lw=1.5)[0] for _ in range(4)],
                 top=axt.plot([], [], color=col, lw=1.5)[0], dot=axt.plot([], [], 'o', color=col, ms=6)[0],
                 ecur=axe.plot([], [], color=col, lw=1.8)[0],
                 hits=ax3.plot([], [], [], 'X', color='red', ms=7, ls='')[0],
                 hitt=axt.plot([], [], 'X', color='red', ms=6, ls='')[0])
        arts.append(a)
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.85)
    tl = ax3.text2D(0.02, 0.02, '', transform=ax3.transAxes, fontsize=12)

    def frame(f):
        k = idx[f]
        for a in arts:
            kk = min(k, len(a['x']) - 1)
            p, Rm = a['x'][kk], a['R'][kk]
            s = slice(max(0, kk - tail), kk + 1)
            a['trail'].set_data_3d(*a['x'][s].T)
            a['path'].set_data_3d(*a['x'][:kk + 1].T)
            tips = p + rot_pts @ Rm.T
            for j, ln in enumerate(a['arms']):
                ln.set_data_3d(*np.stack([tips[2 * j], tips[2 * j + 1]]).T)
            for j, ln in enumerate(a['rot']):
                ln.set_data_3d(*(tips[j] + ring @ Rm.T).T)
            a['top'].set_data(a['x'][:kk + 1, 0], a['x'][:kk + 1, 1])
            a['dot'].set_data([p[0]], [p[1]])
            a['ecur'].set_data(t[:kk + 1], np.maximum(a['err'][:kk + 1], 1e-4))
            h = a['x'][:kk + 1][a['cl'][:kk + 1] < 0]
            if len(h):
                a['hits'].set_data_3d(*h[::5].T)
                a['hitt'].set_data(h[::5, 0], h[::5, 1])
        tl.set_text(f't = {t[k]:.2f} s   ({speed:g}x speed)')
        return []

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    if out.endswith('.gif'):
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000))
    frame(n_frames - 1)
    fig.savefig(os.path.splitext(out)[0] + '_final.png', dpi=130)
    plt.close(fig)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='wind')
    ap.add_argument('--regime', default='obsx')
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--controllers', nargs='+', required=True, help='spec=Label (last one is drawn as ours)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--speed', type=float, default=0.5)
    ap.add_argument('--cache', default='icon_mpc/results/rollouts')
    a = ap.parse_args()
    specs, labels = zip(*[c.rsplit('=', 1) if '=' in c.rsplit(']', 1)[-1] else (c, c) for c in a.controllers])
    runs = simulate(a.experiment, a.regime, a.trial, list(specs), a.cache)
    animate(runs, list(labels), a.out, a.title or f'{a.experiment} / {a.regime} / trial {a.trial}', speed=a.speed)


if __name__ == '__main__':
    main()
