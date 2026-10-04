"""Summarise a stress study: per-regime RMSE / success / tube violation / max tilt, and paired
one-sided Wilcoxon tests of a reference controller against every other controller.
Failures (non-finite RMSE) are ranked worst (set to +inf -> large finite value for the test)."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def short(c):
    return (c.replace('r_v=0.15,r_w=0.15,q_F=10,q_tau=0.1,', '').replace('filt=1,', '').replace(',filt=1', '')
            .replace('filt=1', '').replace('model=', '').replace('[]', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--ref', required=True, help='substring of the reference controller')
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d['ok'] = np.isfinite(d.rmse) & (d.rmse < 1.0)
    d['r'] = np.where(d.ok, d.rmse, 10.0)
    d['c'] = d.controller.map(short)
    ref = [c for c in d.c.unique() if a.ref in c][0]
    pd.set_option('display.width', 250)
    for regime, g in d.groupby('regime', sort=False):
        t = g.groupby('c').agg(rmse_med=('r', 'median'), rmse_ok=('rmse', lambda s: s[np.isfinite(s) & (s < 1)].mean()),
                               ok=('ok', 'mean'), tube=('tube10_viol', 'mean'), tilt=('max_tilt_deg', 'max'),
                               ms=('ctrl_ms', 'mean'))
        piv = g.pivot_table(index=['experiment', 'trial'], columns='c', values='r')
        ps, wins = {}, {}
        for c in piv.columns:
            if c == ref:
                continue
            x, y = piv[ref], piv[c]
            m = x.notna() & y.notna()
            diff = (x - y)[m]
            wins[c] = f"{int((diff < 0).sum())}/{int(m.sum())}"
            ps[c] = wilcoxon(x[m], y[m], alternative='less').pvalue if (diff != 0).any() else 1.0
        t['ref_wins'] = pd.Series(wins)
        t['p(ref<c)'] = pd.Series(ps)
        print(f"\n=== {regime}  (ref = {ref})")
        print(t.sort_values('rmse_med').round(4).to_string())


if __name__ == '__main__':
    main()
