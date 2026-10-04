"""Generate every figure (PDF) and table (LaTeX) of the L4DC paper from the result CSVs.

    python -m icon_mpc.paper_figures            # writes L4DC_paper/figures/*.pdf, L4DC_paper/tables/*.tex
"""
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'icon_mpc', 'results')
PAPER = os.path.join(ROOT, 'L4DC_paper')
FIG, TAB = os.path.join(PAPER, 'figures'), os.path.join(PAPER, 'tables')
FAIL = 10.0
REGIMES = ['base', 'noise', 'heavy', 'outlier', 'dropout', 'flicker', 'impulse', 'stress']
REG_LABEL = {'base': 'Base', 'noise': 'Gauss.\\ noise', 'heavy': 'Heavy-tail', 'outlier': 'Outliers',
             'dropout': 'Dropout', 'flicker': 'Rotor flicker', 'impulse': 'Impacts', 'stress': 'Stress (all)'}
REG_PLAIN = {k: v.replace('\\ ', ' ') for k, v in REG_LABEL.items()}
STOCK = ['geo', 'geo-a', 'l1geo', 'indi-a', 'l1mpc', 'mpc', 'xadap']
STOCK_NAME = {'geo': 'Geometric', 'geo-a': 'Geo-adaptive', 'l1geo': 'L1-Geo', 'indi-a': 'INDI-A',
              'l1mpc': 'L1-MPC', 'mpc': 'MPC', 'xadap': 'XAdap'}
C = dict(stock='#9e9e9e', kf='#ff7f0e', mmae='#9467bd', gdn='#2ca02c', mamba='#1f77b4', nom='#8c564b',
         agg='#17becf', gru='#e377c2')

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'font.family': 'serif', 'pdf.fonttype': 42, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})


def cname(c):
    m = re.search(r'model=([a-z]+)_mix2(_s\d)?', c)
    if c.startswith('nmpc+agg'):
        return 'agg'
    if m:
        return f'{m.group(1)}' + (m.group(2) or '_s0')
    if c.startswith('nmpc+kf'):
        return 'kf'
    if c.startswith('nmpc+mmae'):
        return 'mmae'
    return c


def load_main():
    d = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in ('final_baselines.csv', 'final_learned.csv')],
                  ignore_index=True)
    d['c'] = d.controller.map(cname)
    d['ok'] = np.isfinite(d.rmse) & (d.rmse < 1.0)
    d['r'] = np.where(d.ok, d.rmse, FAIL)
    rows = [d]
    for arch in ('gdn', 'mamba'):
        s = d[d.c.str.startswith(f'{arch}_s')]
        g = s.groupby(['regime', 'experiment', 'trial'], as_index=False).agg(
            r=('r', 'mean'), ok=('ok', 'mean'), ctrl_ms=('ctrl_ms', 'mean'), tube10_viol=('tube10_viol', 'mean'))
        g['c'] = arch
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def fmt_p(p):
    if p < 1e-6:
        return '$<\\!10^{-6}$'
    e = int(np.floor(np.log10(p)))
    return f'{p:.2f}' if p >= 0.01 else f'${p / 10 ** e:.0f}\\!\\cdot\\!10^{{{e}}}$'


def main_table_and_fig(d):
    rows, bars = [], []
    for reg in REGIMES:
        g = d[d.regime == reg]
        piv = g.pivot_table(index=['experiment', 'trial'], columns='c', values='r')
        ok = g.groupby('c').ok.mean()
        med = piv.median()
        best_s = med[STOCK].idxmin()
        best_b = med[['kf', 'mmae']].idxmin()

        def test(a, b):
            m = piv[a].notna() & piv[b].notna()
            x, y = piv[a][m], piv[b][m]
            p = wilcoxon(x, y, alternative='less').pvalue if np.any(x != y) else 1.0
            return int((x < y).sum()), int(m.sum()), p
        ws, n, ps = test('gdn', best_s)
        wb, _, pb = test('gdn', best_b)
        rows.append(dict(reg=reg, s=best_s, s_r=med[best_s], s_ok=ok[best_s], kf=med['kf'], kf_ok=ok['kf'],
                         mm=med['mmae'], mm_ok=ok['mmae'], gdn=med['gdn'], gdn_ok=ok['gdn'], mamba=med['mamba'],
                         mamba_ok=ok['mamba'], gain_s=med[best_s] / med['gdn'], ws=ws, n=n, ps=ps,
                         gain_b=med[best_b] / med['gdn'], wb=wb, pb=pb, b=best_b))
        q = lambda c: (piv[c].quantile(0.25), piv[c].median(), piv[c].quantile(0.75))
        bars.append({k: q(k) for k in [best_s, 'kf', 'mmae', 'mamba', 'gdn']} | {'stock': best_s})
    t = pd.DataFrame(rows)

    def cell(r, ok):
        s = f'{100 * r:.2f}' if r < 1 else 'fail'
        return s + ('' if ok > 0.995 else f'$^{{\\,{100 * ok:.0f}\\%}}$')
    L = ['\\begin{tabular}{l l r r r r r r r}', '\\toprule',
         ' & \\multicolumn{2}{c}{Best stock controller} & NMPC+ & NMPC+ & \\multicolumn{2}{c}{\\textbf{ICON-MPC (ours)}} & '
         '\\multicolumn{2}{c}{GDN vs.\\ best stock} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}',
         'Regime & name & RMSE & rob.\\ KF & MMAE & Mamba & GDN & gain & wins \\\\', '\\midrule']
    for r in t.itertuples():
        best_ours = min(r.gdn, r.mamba, r.kf, r.mm)
        b = lambda v, s: f'\\textbf{{{s}}}' if v == best_ours else s
        L.append(f'{REG_LABEL[r.reg]} & {STOCK_NAME[r.s]} & {cell(r.s_r, r.s_ok)} & {b(r.kf, cell(r.kf, r.kf_ok))} & '
                 f'{b(r.mm, cell(r.mm, r.mm_ok))} & {b(r.mamba, cell(r.mamba, r.mamba_ok))} & '
                 f'{b(r.gdn, cell(r.gdn, r.gdn_ok))} & {r.gain_s:.1f}$\\times$ & {r.ws}/{r.n} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'main_results.tex'), 'w').write('\n'.join(L) + '\n')

    # vs strong (our own) baselines
    L = ['\\begin{tabular}{l l r r r}', '\\toprule',
         'Regime & best strong baseline & GDN gain & wins & $p$ (Wilcoxon) \\\\', '\\midrule']
    for r in t.itertuples():
        L.append(f"{REG_LABEL[r.reg]} & {'robust KF' if r.b == 'kf' else 'MMAE'} & {r.gain_b:.2f}$\\times$ & "
                 f'{r.wb}/{r.n} & {fmt_p(r.pb)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'strong_baselines.tex'), 'w').write('\n'.join(L) + '\n')

    # figure: grouped bars (median, IQR whiskers), log scale
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    keys = [('stock', 'best stock', C['stock']), ('kf', 'NMPC + robust KF', C['kf']), ('mmae', 'NMPC + MMAE', C['mmae']),
            ('mamba', 'ICON-MPC (Mamba)', C['mamba']), ('gdn', 'ICON-MPC (GDN)', C['gdn'])]
    w = 0.16
    for j, (k, lab, col) in enumerate(keys):
        for i, b in enumerate(bars):
            lo, m, hi = b[b['stock']] if k == 'stock' else b[k]
            m, lo, hi = min(m, 1.5), min(lo, 1.5), min(hi, 1.5)
            ax.bar(i + (j - 2) * w, m, w, color=col, label=lab if i == 0 else None, edgecolor='k', lw=0.3)
            ax.plot([i + (j - 2) * w] * 2, [lo, hi], color='k', lw=0.6)
            if k == 'stock':
                ax.text(i + (j - 2) * w, min(hi, 1.4) * 1.15, STOCK_NAME[b['stock']], rotation=90, ha='center',
                        va='bottom', fontsize=5.5, color='0.3')
    ax.set_yscale('log')
    ax.set_ylim(2e-3, 3)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([REG_PLAIN[r].replace(' ', '\n', 1) for r in REGIMES])
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax.set_ylabel('position RMSE [m]')
    fig.legend(ncol=5, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, 'tracking_results.pdf'), bbox_inches='tight')
    plt.close(fig)
    return t


def seeds_table(d):
    g = d[d.c.str.contains('_s')].groupby(['regime', 'c']).r.median().unstack() * 100
    cols = ['gdn_s0', 'gdn_s1', 'gdn_s2', 'mamba_s0', 'mamba_s1', 'mamba_s2']
    L = ['\\begin{tabular}{l rrr rrr}', '\\toprule',
         ' & \\multicolumn{3}{c}{GDN seed} & \\multicolumn{3}{c}{Mamba seed} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}', 'Regime & 0 & 1 & 2 & 0 & 1 & 2 \\\\', '\\midrule']
    for reg in REGIMES:
        L.append(REG_LABEL[reg] + ' & ' + ' & '.join(f'{g.loc[reg, c]:.2f}' for c in cols) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'seeds.tex'), 'w').write('\n'.join(L) + '\n')


def compute_table(d):
    keep = STOCK + ['kf', 'mmae', 'gdn_s0', 'mamba_s0']
    g = d[d.c.isin(keep)]
    t = g.groupby('c').agg(ms=('ctrl_ms', 'mean'), p99=('ctrl_p99_ms', 'mean'))
    t['ok'] = g[g.regime == 'stress'].groupby('c').ok.mean()
    t['r'] = g[g.regime == 'stress'].groupby('c').r.median()
    lab = dict(STOCK_NAME, kf='NMPC + robust KF', mmae='NMPC + MMAE', gdn_s0='\\textbf{ICON-MPC (GDN)}',
               mamba_s0='ICON-MPC (Mamba)')
    L = ['\\begin{tabular}{l r r r r}', '\\toprule',
         'Controller & mean [ms] & p99 [ms] & stress success & stress RMSE [cm] \\\\', '\\midrule']
    for c in keep:
        r = t.loc[c]
        rm = f'{100 * r.r:.1f}' if r.r < 1 else 'fail'
        L.append(f'{lab[c]} & {r.ms:.2f} & {r.p99:.1f} & {100 * r.ok:.0f}\\% & {rm} \\\\')
        if c == STOCK[-1]:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'compute.tex'), 'w').write('\n'.join(L) + '\n')
    return t


def obstacle_figs():
    v1 = pd.read_csv(os.path.join(RES, 'obstacles_v1.csv'))
    mg = [pd.read_csv(os.path.join(RES, f)) for f in ('obstacles_margin.csv', 'obstacles_aci.csv', 'obstacles_aci_baselines.csv', 'obstacles_aci_baselines2.csv')
          if os.path.exists(os.path.join(RES, f))]
    d = pd.concat([v1, *mg]).drop_duplicates(['regime', 'experiment', 'trial', 'controller'])
    d['ok'] = np.isfinite(d.rmse) & (d.rmse < 1.0)
    d['unsafe'] = ((d.collision > 0) | ~d.ok).astype(int)
    F = 'filt=1,gate=1,ffu=1,qyaw=1'
    K = f'nmpc+kf[r_v=0.15,r_w=0.15,q_F=10,q_tau=1,{F},obs=1'
    M = 'nmpc+mmae[filt=1,forget=0.95,gate=1,ffu=1,qyaw=1,obs=1'
    G = f'nmpc+learned[model=gdn_mix2,{F},obs=1'
    N = 'nmpc+nominal[filt=1,gate=1,qyaw=1,obs=1'
    ctrls = [('stock', 'best stock (no constraints)', C['stock']),
             (f'{N},obsm=0.03]', 'NMPC nominal + 3 cm', C['nom']),
             (f'{K}]', 'NMPC+KF, no margin', '#ffbb78'),
             (f'{K},obsm=0.03]', 'NMPC+KF + 3 cm', C['kf']),
             (f'{K},aci=1]', 'NMPC+KF + ACI', '#ff9896'),
             (f'{K},aci=1,obsm=0.01]', 'NMPC+KF + ACI + 1 cm', '#d62728'),
             (f'{M},obsm=0.03]', 'NMPC+MMAE + 3 cm', '#c5b0d5'),
             (f'{M},aci=1,obsm=0.01]', 'NMPC+MMAE + ACI + 1 cm', C['mmae']),
             (f'{G},obsm=0.03]', 'ICON + 3 cm', '#98df8a'),
             (f'{G},aci=1,obsm=0.01]', 'ICON + ACI (ours)', C['gdn'])]
    regs = ['obsx', 'obsx_noise', 'obsx_flicker', 'obsx_stress']
    rl = {'obsx': 'Base', 'obsx_noise': 'Noise', 'obsx_flicker': 'Rotor flicker', 'obsx_stress': 'Stress'}
    stats = {}
    for reg in regs:
        g = d[d.regime == reg]
        st = g[g.controller.isin(STOCK)].groupby('controller').unsafe.mean()
        for k, _, _ in ctrls:
            h = g[g.controller == st.idxmin()] if k == 'stock' else g[g.controller == k]
            if len(h):
                stats[(reg, k)] = (h.unsafe.mean(), h.rmse[h.ok].median(), h.min_clear.replace(-np.inf, np.nan).median(),
                                   len(h), h.coll_frac.mean())
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.1), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axs[0]
    w = 0.085
    for j, (k, lab, col) in enumerate(ctrls):
        for i, reg in enumerate(regs):
            if (reg, k) in stats:
                v = stats[(reg, k)][0]
                ax.bar(i + (j - 4.5) * w, 100 * v, w, color=col, edgecolor='k', lw=0.3, label=lab if i == 0 else None)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([rl[r] for r in regs])
    ax.set_ylabel('unsafe trials [%]'); ax.set_yscale('symlog', linthresh=2); ax.set_ylim(0, 130)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(['0', '1', '2', '5', '10', '20', '50', '100'])
    ax = axs[1]
    for k, lab, col in ctrls[1:]:
        if ('obsx_stress', k) in stats:
            u, r, *_ = stats[('obsx_stress', k)]
            ax.scatter(100 * r, 100 * u, color=col, s=30, edgecolor='k', lw=0.4, zorder=3)
    ax.set_xlabel('median RMSE, stress [cm]'); ax.set_ylabel('unsafe trials, stress [%]')
    ax.set_title('safety vs. tracking (stress)', fontsize=7)
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, fontsize=5.5, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(FIG, 'obstacle_safety.pdf'), bbox_inches='tight')
    plt.close(fig)

    # table with McNemar tests vs ours
    ref = f'{G},aci=1,obsm=0.01]'
    L = ['\\begin{tabular}{l rrr rrr}', '\\toprule',
         ' & \\multicolumn{3}{c}{Base (no sensing faults)} & \\multicolumn{3}{c}{Stress} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
         'Controller & unsafe & RMSE & $p$ & unsafe & RMSE & $p$ \\\\', '\\midrule']
    key = ['experiment', 'trial']
    for k, lab, _ in ctrls:
        cells = []
        for reg in ('obsx', 'obsx_stress'):
            if (reg, k) not in stats:
                cells += ['--'] * 3
                continue
            u, r, *_ = stats[(reg, k)]
            g = d[d.regime == reg]
            if k == 'stock':
                st = g[g.controller.isin(STOCK)].groupby('controller').unsafe.mean()
                kk = st.idxmin()
            else:
                kk = k
            if kk == ref:
                p = '--'
            else:
                j = g[g.controller == ref].set_index(key).unsafe.to_frame('a').join(
                    g[g.controller == kk].set_index(key).unsafe.to_frame('b'), how='inner')
                b, c = int(((j.b == 1) & (j.a == 0)).sum()), int(((j.b == 0) & (j.a == 1)).sum())
                p = fmt_p(binomtest(b, b + c, 0.5, alternative='greater').pvalue) if b + c else '1.00'
            cells += [f'{100 * u:.1f}\\%', f'{100 * r:.1f}', p]
        name = f'\\textbf{{{lab}}}' if k == ref else lab
        L.append(name + ' & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'obstacles.tex'), 'w').write('\n'.join(L) + '\n')
    return stats


def margin_fig():
    import glob
    cache = os.path.join(RES, 'rollouts')
    f = [p for p in glob.glob(os.path.join(cache, 'wind_obsx_stress_5_*.npz')) if 'aci=1' in p and 'learned' in p]
    if not f:
        return
    r = np.load(f[0], allow_pickle=True)
    others = {'mpc': ('stock MPC', C['stock']),
              'nmpc+kf[r_v=0.15,r_w=0.15,q_F=10,q_tau=1,filt=1,gate=1,ffu=1,qyaw=1,obs=1]': ('NMPC+KF, no margin', '#ffbb78')}
    obs = r['obstacles']
    obs = obs[obs[:, 2] > 0]

    def clear(x):
        return np.min(np.linalg.norm(x[:, None, :2] - obs[None, :, :2], axis=-1) - obs[None, :, 2], axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(3.1, 2.4), sharex=True)
    ax = axs[0]
    for k, (lab, col) in others.items():
        p = os.path.join(cache, f'wind_obsx_stress_5_{k}.npz')
        if os.path.exists(p):
            o = np.load(p, allow_pickle=True)
            ax.plot(o['time'], 100 * clear(o['x']), color=col, lw=0.9, label=lab)
    ax.plot(r['time'], 100 * clear(r['x']), color=C['gdn'], lw=1.1, label='ICON + ACI (ours)')
    ax.axhline(0, color='r', lw=0.7)
    ax.set_ylim(-15, 30); ax.set_ylabel('clearance [cm]')
    ax.legend(fontsize=5.5, frameon=False, ncol=1, loc='upper right')
    ax = axs[1]
    m = r['margin']
    if len(m):
        ax.plot(m[:, 0], 100 * m[:, 1], color=C['gdn'], lw=1)
        deg = m[:, 2] > 0
        ax.fill_between(m[:, 0], 0, 100 * m[:, 1].max() * 1.1, where=deg, color='r', alpha=0.15, lw=0,
                        label='degraded sensing')
    ax.set_ylabel('margin [cm]'); ax.set_xlabel('t [s]')
    ax.legend(fontsize=5.5, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'aci_margin.pdf'), bbox_inches='tight')
    plt.close(fig)


def ablation_table(d_main):
    p = os.path.join(RES, 'ablation.csv')
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    a['ok'] = np.isfinite(a.rmse) & (a.rmse < 1.0)
    a['r'] = np.where(a.ok, a.rmse, FAIL)
    F = 'filt=1,gate=1,ffu=1,qyaw=1'
    full = d_main[d_main.c == 'gdn_s0'][['regime', 'experiment', 'trial', 'r', 'ok']].assign(v='full')
    kf = d_main[d_main.c == 'kf'][['regime', 'experiment', 'trial', 'r', 'ok']].assign(v='kf')
    mam = d_main[d_main.c == 'mamba_s0'][['regime', 'experiment', 'trial', 'r', 'ok']].assign(v='mamba')
    names = {f'nmpc+nominal[{F}]': 'nominal', f'nmpc+aero[{F}]': 'aero',
             'nmpc+learned[model=gdn_mix2,filt=1,gate=0,ffu=1,qyaw=1]': 'nogate',
             'nmpc+learned[model=gdn_mix2,filt=0,gate=1,ffu=1,qyaw=1]': 'nofilt',
             'nmpc+learned[model=gdn_mix2,filt=1,gate=1,ffu=0,qyaw=1]': 'noffu',
             f'nmpc+learned[model=gru_mix2,{F}]': 'gru'}
    a['v'] = a.controller.map(names)
    allv = pd.concat([full, kf, mam, a[['regime', 'experiment', 'trial', 'r', 'ok', 'v']].dropna(subset=['v'])])
    rows = [('nominal', 'NMPC, nominal model (no drag, no estimator)'), ('aero', '\\quad + rotor-drag model'),
            ('kf', '\\quad + robust lumped KF'), ('gru', '\\quad + learned estimator: GRU'),
            ('mamba', '\\quad + learned estimator: Mamba'), ('full', '\\quad + learned estimator: GDN (full)'),
            ('nogate', 'full $-$ robust front end'), ('nofilt', 'full $-$ learned state filter'),
            ('noffu', 'full $-$ offset-free input reference')]
    regs = REGIMES
    L = ['\\begin{tabular}{l ' + 'r' * len(regs) + '}', '\\toprule',
         'Variant & ' + ' & '.join(REG_LABEL[r] for r in regs) + ' \\\\', '\\midrule']
    for v, lab in rows:
        g = allv[allv.v == v]
        if not len(g):
            continue
        cells = []
        for reg in regs:
            h = g[g.regime == reg]
            if not len(h):
                cells.append('--'); continue
            med, ok = h.r.median(), h.ok.mean()
            s = f'{100 * med:.2f}' if med < 1 else 'fail'
            cells.append(s + ('' if ok > 0.995 else f'$^{{{100 * ok:.0f}}}$'))
        L.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
        if v in ('full',):
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(TAB, 'ablation.tex'), 'w').write('\n'.join(L) + '\n')


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    d = load_main()
    t = main_table_and_fig(d)
    print(t.round(4).to_string())
    print(compute_table(d).round(3).to_string())
    seeds_table(d)
    st = obstacle_figs()
    for k, v in st.items():
        print(k[0], k[1][-40:], np.round(v, 4))
    margin_fig()
    ablation_table(d)
    for src, dst in (('safety_obsx_stress_wind_t5_final.png', 'snapshot_safety.png'),
                     ('tracking_stress_force_t16_final.png', 'snapshot_tracking.png')):
        p = os.path.join(RES, 'media', src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(FIG, dst))


if __name__ == '__main__':
    main()
