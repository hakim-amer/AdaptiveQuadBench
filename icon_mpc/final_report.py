"""Final comparison table: ours (per training seed and seed-mean) vs every baseline, per regime.

Failures (non-finite RMSE or RMSE > 1 m) are ranked worst in the paired tests. The reference for
significance is the *best baseline in that regime* (lowest median), chosen post hoc, which is
conservative for us.

    python -m icon_mpc.final_report icon_mpc/results/final_baselines.csv icon_mpc/results/final_learned.csv
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FAIL = 10.0


def name(c):
    m = re.search(r'model=([a-z]+)_mix2(_s\d)?', c)
    if c.startswith('nmpc+agg'):
        return 'OURS-agg(KF+GDN)'
    if m:
        return f'OURS-{m.group(1)}' + (m.group(2) or '_s0')
    if c.startswith('nmpc+kf'):
        return 'NMPC+robustKF'
    if c.startswith('nmpc+mmae'):
        return 'NMPC+MMAE'
    return c


def main():
    d = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
    d['c'] = d.controller.map(name)
    d['ok'] = np.isfinite(d.rmse) & (d.rmse < 1.0)
    d['r'] = np.where(d.ok, d.rmse, FAIL)
    # seed-averaged learned methods (per trial mean over the 3 training seeds)
    for arch in ('gdn', 'mamba'):
        s = d[d.c.str.startswith(f'OURS-{arch}_s')]
        if s.c.nunique() > 1:
            g = s.groupby(['regime', 'experiment', 'trial'], as_index=False).agg(
                r=('r', 'mean'), ok=('ok', 'mean'), tube10_viol=('tube10_viol', 'mean'),
                max_tilt_deg=('max_tilt_deg', 'mean'), heading_deg=('heading_deg', 'mean'),
                ctrl_ms=('ctrl_ms', 'mean'))
            g['c'] = f'OURS-{arch}(3-seed mean)'
            d = pd.concat([d, g], ignore_index=True)
    ours = [c for c in d.c.unique() if c.startswith('OURS')]
    base = [c for c in d.c.unique() if not c.startswith('OURS')]
    pd.set_option('display.width', 250)
    summary = []
    for regime, g in d.groupby('regime', sort=False):
        piv = g.pivot_table(index=['experiment', 'trial'], columns='c', values='r')
        med = piv.median()
        best_b = med[base].idxmin()
        t = g.groupby('c').agg(rmse_med=('r', 'median'), ok=('ok', 'mean'), tube=('tube10_viol', 'mean'),
                               heading=('heading_deg', 'median'), ms=('ctrl_ms', 'mean'))
        for c in ours:
            m = piv[c].notna() & piv[best_b].notna()
            x, y = piv[c][m], piv[best_b][m]
            t.loc[c, 'wins_vs_best_base'] = f'{int((x < y).sum())}/{int(m.sum())}'
            t.loc[c, 'p'] = wilcoxon(x, y, alternative='less').pvalue
            t.loc[c, 'gain_x'] = y.median() / x.median()
        print(f'\n=== {regime}   (best baseline: {best_b})')
        print(t.sort_values('rmse_med').round(4).to_string())
        r0 = t.loc['OURS-gdn(3-seed mean)'] if 'OURS-gdn(3-seed mean)' in t.index else None
        if r0 is not None:
            summary.append((regime, best_b, t.loc[best_b, 'rmse_med'], t.loc[best_b, 'ok'],
                            r0.rmse_med, r0.ok, r0.gain_x, r0.p))
    print('\n=== SUMMARY  ours = GDN (3-seed mean) vs best baseline per regime')
    print(pd.DataFrame(summary, columns=['regime', 'best_baseline', 'base_rmse', 'base_ok', 'ours_rmse',
                                         'ours_ok', 'gain_x', 'p']).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
