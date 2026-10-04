"""Obstacle-avoidance safety analysis: collision rate, min clearance, tracking error, per regime.
Paired tests vs a reference controller: exact McNemar on collisions, Wilcoxon on min clearance."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def short(c):
    return (c.replace('r_v=0.15,r_w=0.15,q_F=10,q_tau=1,', '').replace('filt=1,gate=1,ffu=1,qyaw=1', 'F')
             .replace('filt=1,forget=0.95,gate=1,ffu=1,qyaw=1', 'F').replace('filt=1,gate=1,qyaw=1', 'F'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--ref', default='gdn_mix2,filt=1,gate=1,ffu=1,qyaw=1,obs=1,aci=1]')
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in a.csv], ignore_index=True)
    d = d.drop_duplicates(['regime', 'experiment', 'trial', 'controller'])
    d['ok'] = np.isfinite(d.rmse) & (d.rmse < 1.0)
    d['unsafe'] = ((d.collision > 0) | ~d.ok).astype(int)  # a crash counts as unsafe
    key = ['experiment', 'trial']
    for reg, g in d.groupby('regime'):
        ref = g[g.controller.str.contains(a.ref, regex=False)].set_index(key)
        rows = []
        for c, h in g.groupby('controller'):
            h = h.set_index(key)
            j = ref.join(h, how='inner', lsuffix='_r')
            row = dict(controller=short(c), n=len(h), unsafe=h.unsafe.mean(), coll_time=h.coll_frac.mean(),
                       min_clear_med=h.min_clear.replace(-np.inf, np.nan).median(),
                       rmse_med=h.rmse[h.ok].median(), ok=h.ok.mean(), ms=h.ctrl_ms.mean())
            if len(j) and not h.controller.iloc[0] == ref.controller.iloc[0]:
                b = int(((j.unsafe == 1) & (j.unsafe_r == 0)).sum())  # ref safe, other unsafe
                cc = int(((j.unsafe == 0) & (j.unsafe_r == 1)).sum())
                row['ref_safer'] = f'{b}:{cc}'
                row['p_mcnemar'] = binomtest(b, b + cc, 0.5, alternative='greater').pvalue if b + cc else 1.0
                mc, mr = j.min_clear.replace(-np.inf, -1).values, j.min_clear_r.replace(-np.inf, -1).values
                row['p_clear'] = wilcoxon(mr, mc, alternative='greater').pvalue if np.any(mr != mc) else 1.0
            rows.append(row)
        t = pd.DataFrame(rows).set_index('controller').sort_values('unsafe')
        print(f'\n=== {reg}  (ref = {short(a.ref)})')
        print(t.to_string(float_format=lambda x: f'{x:.4g}'))


if __name__ == '__main__':
    main()
