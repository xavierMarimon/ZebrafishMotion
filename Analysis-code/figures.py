"""Vector figures for the manuscript."""
import warnings
warnings.filterwarnings('ignore')
import json
from math import factorial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nld
import nld2
import pipeline as P

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7,
                     'ytick.major.width': 0.7, 'savefig.bbox': 'tight',
                     'axes.titlesize': 9, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'figure.dpi': 300})

CH, CS = '#2c6fb5', '#d1495b'          # healthy / scoliotic
TRAJ = ("/sessions/tender-affectionate-carson/mnt/Article Zebrafish/"
        "zebrafish material and code/TRAJECTORY RESULTS")

d = pd.read_csv('features_all11.csv')
s = pd.read_csv('surrogates11.csv')
d['grp'] = np.where(d.fish.str.startswith('sf'), 'scoliotic', 'healthy')
s['grp'] = np.where(s.fish.str.startswith('sf'), 'scoliotic', 'healthy')
trials = {(f, t): h for f, t, h in P.find_trials(TRAJ)}


# ---------------------------------------------------------------- bounds
def hc_bounds(n, npts=300):
    """Envelope of the complexity-entropy plane (Martin et al. 2006)."""
    Q0 = -2.0 / (((n + 1.0) / n) * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))

    def HC(P):
        P = np.asarray(P, float)
        P = P / P.sum()
        S = nld2._shannon
        H = S(P) / np.log(n)
        Pe = np.full(n, 1.0 / n)
        JS = S(0.5 * (P + Pe)) - 0.5 * S(P) - 0.5 * S(Pe)
        return H, Q0 * JS * H

    pts = []
    for j in range(0, n - 1):                      # j zeroed states
        k = n - j
        for p in np.linspace(1.0 / k, 1.0, npts):
            P = np.zeros(n)
            P[0] = p
            if k > 1:
                P[1:k] = (1 - p) / (k - 1)
            pts.append(HC(P))
    pts = np.array(pts)
    o = np.argsort(pts[:, 0])
    pts = pts[o]
    grid = np.linspace(0, 1, 250)
    idx = np.digitize(pts[:, 0], grid)
    up, lo = [], []
    for g in range(1, len(grid)):
        m = pts[idx == g]
        if len(m):
            up.append((grid[g], m[:, 1].max()))
            lo.append((grid[g], m[:, 1].min()))
    return np.array(lo), np.array(up)


# ---------------------------------------------------------------- fig 1
def fig_complexity_plane(fname='fig_hc_plane.pdf'):
    n = factorial(5)
    lo, up = hc_bounds(n)
    rng = np.random.default_rng(5)
    N = 6000
    refs = {}
    x = np.empty(N); x[0] = 0.2
    for i in range(1, N):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    refs['Logistic map'] = x
    hx, hy, H = 0.1, 0.3, []
    for i in range(N + 500):
        hx, hy = 1 - 1.4 * hx ** 2 + hy, 0.3 * hx
        H.append(hx)
    refs['Hénon map'] = np.array(H[500:])
    refs['White noise'] = rng.standard_normal(N)
    f = np.fft.rfftfreq(N)
    for beta, nm in ((0.5, r'$1/f$ noise'), (1.0, r'$1/f^{2}$ noise')):
        S = np.ones_like(f); S[1:] = f[1:] ** -beta
        refs[nm] = np.fft.irfft(S * np.exp(2j * np.pi * rng.random(len(f))), N)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.fill_between(up[:, 0], lo_i := np.interp(up[:, 0], lo[:, 0], lo[:, 1]),
                    up[:, 1], color='0.93', lw=0, zorder=0)
    ax.plot(up[:, 0], up[:, 1], color='0.55', lw=0.7, zorder=1)
    ax.plot(lo[:, 0], lo[:, 1], color='0.55', lw=0.7, zorder=1)

    for nm, v in refs.items():
        Hh, Cc = nld2.statistical_complexity(v, d=5)
        mk = 'D' if 'noise' in nm else 's'
        col = '#666666' if 'noise' in nm else '#e08214'
        ax.scatter(Hh, Cc, marker=mk, s=26, c=col, ec='k', lw=0.4, zorder=4)
        offs = {'Logistic map': (-16, 6), 'H\u00e9non map': (-4, -13),
                'White noise': (-9, 4), r'$1/f$ noise': (-9, 6),
                r'$1/f^{2}$ noise': (-9, 3)}
        has = {'White noise': 'right', r'$1/f$ noise': 'right',
               r'$1/f^{2}$ noise': 'right'}
        ax.annotate(nm, (Hh, Cc), textcoords='offset points',
                    xytext=offs.get(nm, (0, 7)),
                    ha=has.get(nm, 'center'), fontsize=6.4)

    g = d.groupby('fish')[['PE_H5', 'PE_C5']].mean()
    for f_, row in g.iterrows():
        sc = f_.startswith('sf')
        ax.scatter(row.PE_H5, row.PE_C5, s=30, marker='o' if not sc else '^',
                   c=CS if sc else CH, ec='k', lw=0.4, zorder=5)
    ax.annotate('zebrafish\nbending angle', (g.PE_H5.mean(), g.PE_C5.mean()),
                textcoords='offset points', xytext=(-16, -26), ha='center',
                fontsize=6.8, fontweight='bold',
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.set_xlabel(r'normalised permutation entropy $H$')
    ax.set_ylabel(r'statistical complexity $C_{JS}$')
    ax.set_xlim(0, 1.06); ax.set_ylim(0, 0.50)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc=CH, mec='k',
                              ms=5, label='healthy'),
                       Line2D([], [], marker='^', ls='', mfc=CS, mec='k',
                              ms=5, label='scoliotic')],
              loc='upper left', frameon=False, borderaxespad=0.2)
    fig.savefig(fname); plt.close(fig)
    print('ok', fname)


# ---------------------------------------------------------------- fig 2
def fig_surrogates(fname='fig_surrogates.pdf'):
    ST = ['SampEn', 'PE_H5', 'ENTR', 'DET', 'Trev1', 'PE_missing5']
    LB = [r'SampEn', r'$H$ (perm. entropy)', r'RQA $ENTR$', r'RQA $DET$',
          r'$T_{rev}$', 'forbidden patterns']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    pos = np.arange(len(ST))
    for i, k in enumerate(ST):
        for grp, col, off in (('healthy', CH, -0.16), ('scoliotic', CS, 0.16)):
            z = s.loc[s.grp == grp, 'z_' + k].values
            ax.scatter(np.full(len(z), pos[i] + off)
                       + np.random.default_rng(i).normal(0, .04, len(z)),
                       z, s=5, c=col, alpha=.55, lw=0)
            ax.plot([pos[i] + off - .1, pos[i] + off + .1],
                    [np.mean(z)] * 2, color='k', lw=1.2, zorder=5)
    ax.axhline(0, color='k', lw=.6)
    for y in (-2, 2):
        ax.axhline(y, color='0.5', lw=.6, ls='--')
    ax.set_xticks(pos); ax.set_xticklabels(LB, rotation=30, ha='right')
    ax.set_ylabel(r'$z$ vs IAAFT surrogates')
    ax.set_title('Deviation from the linear null', loc='left')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=CH, ms=4,
                              label='healthy'),
                       Line2D([], [], marker='o', ls='', color=CS, ms=4,
                              label='scoliotic')],
              loc='lower left', frameon=False, ncol=2)

    ax = axes[1]
    frac = [(s['p_' + k] <= 0.05).mean() * 100 for k in ST]
    ax.barh(pos, frac, color='#7fa8d0', ec='k', lw=.5)
    ax.set_yticks(pos); ax.set_yticklabels(LB)
    ax.set_xlabel('trials rejecting the linear null (%)')
    ax.set_xlim(0, 100)
    for i, v in enumerate(frac):
        ax.text(v + 1.5, pos[i], f'{v:.0f}', va='center', fontsize=6.5)
    ax.set_title('Rejection rate (n = 110 trials)', loc='left')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print('ok', fname)


# ---------------------------------------------------------------- fig 3
def fig_no_attractor(fname='fig_no_attractor.pdf'):
    """Evidence that classical attractor invariants are not defined here."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    rng = np.random.default_rng(0)

    # (a) FNN
    ax = axes[0]
    ex = [('f2', 'f2v1'), ('f5', 'f5v3'), ('sf2', 'sf2v1'), ('sf3', 'sf3v4')]
    for f_, t_ in ex:
        pts, fs, _ = P.load_points(trials[(f_, t_)])
        th = P.bending_angle(pts, fs)
        tau = max(1, nld.first_min_tau(nld.average_mutual_information(th, 150)))
        fr = nld.false_nearest_neighbours(th, tau, max_m=8, nref=400,
                                          theiler=int(fs / 10))
        ax.plot(range(1, 9), 100 * fr, '-o', ms=2.5, lw=.9,
                color=CS if f_.startswith('sf') else CH, alpha=.85)
    N = 4000
    x = np.empty(N); x[0] = .2
    for i in range(1, N):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    ax.plot(range(1, 9), 100 * nld.false_nearest_neighbours(x, 1, 8, nref=400),
            '-s', ms=2.5, lw=.9, color='#e08214', label='logistic map')
    ax.axhline(1, color='0.4', ls='--', lw=.7)
    ax.set_xlabel('embedding dimension $m$')
    ax.set_ylabel('false nearest neighbours (%)')
    ax.set_title('(a) FNN does not converge', loc='left')
    ax.legend(frameon=False, loc='upper right')

    # (b) correlation sum local slope
    ax = axes[1]
    for f_, t_ in ex:
        pts, fs, _ = P.load_points(trials[(f_, t_)])
        th = P.bending_angle(pts, fs)
        tau = max(1, nld.first_min_tau(nld.average_mutual_information(th, 150)))
        X = nld.embed(th, 5, tau)
        rs, C = nld.correlation_sum(X, theiler=int(fs / 2), nref=450)
        if rs is None:
            continue
        lr, lC = np.log(rs), np.log(C)
        ax.plot(rs / th.std(), np.gradient(lC, lr), lw=.9,
                color=CS if f_.startswith('sf') else CH, alpha=.85)
    Xl = nld.embed(x, 3, 1)
    rs, C = nld.correlation_sum(Xl, theiler=10, nref=450)
    ax.plot(rs / x.std(), np.gradient(np.log(C), np.log(rs)), lw=.9,
            color='#e08214')
    ax.axhspan(0.9, 1.1, color='#e08214', alpha=.12, lw=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/\sigma$')
    ax.set_ylabel(r'$d\log C(r)/d\log r$')
    ax.set_title('(b) no scaling plateau', loc='left')

    # (c) MSE
    ax = axes[2]
    mse = np.load('mse_curves11.npy')
    sc = d.fish.str.startswith('sf').values
    sca = np.arange(1, mse.shape[1] + 1)
    for m_, c in ((mse[~sc], CH), (mse[sc], CS)):
        mu, sd = np.nanmean(m_, 0), np.nanstd(m_, 0)
        ax.plot(sca, mu, color=c, lw=1.2)
        ax.fill_between(sca, mu - sd, mu + sd, color=c, alpha=.18, lw=0)
    wn = nld2.rcmse(rng.standard_normal(3000), scales=list(range(1, 16)))
    ax.plot(sca, [wn[k] for k in sorted(wn)], color='0.45', lw=1.0, ls='--')
    ax.text(13.5, wn[13] + .30, 'white noise', fontsize=6.2, color='0.35',
            ha='right')
    ax.set_xlabel('scale factor')
    ax.set_ylabel('sample entropy')
    ax.set_title('(c) multiscale entropy', loc='left')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print('ok', fname)


# ---------------------------------------------------------------- fig 4
def fig_signal(fname='fig_signal.pdf'):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6))
    for ax, (f_, t_) in zip(axes[:, 0], [('f2', 'f2v1'), ('sf3', 'sf3v1')]):
        pts, fs, _ = P.load_points(trials[(f_, t_)])
        th = P.bending_angle(pts, fs)
        tt = np.arange(len(th)) / fs
        col = CS if f_.startswith('sf') else CH
        sl = (tt >= 5) & (tt <= 15)
        ax.plot(tt[sl], th[sl], lw=1.1, color=col)
        ax.axhline(np.median(th), color='k', lw=.5, ls='--')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.set_ylim(-180, 180)
        ax.set_title('(%s) %s (%s)'
                     % ('c' if f_.startswith('sf') else 'a',
                        'scoliotic' if f_.startswith('sf') else 'healthy', f_),
                     loc='left')
    axes[1, 0].set_xlabel('time (s)')
    axes[0, 0].set_xticklabels([])

    ax = axes[0, 1]
    for grp, col in (('healthy', CH), ('scoliotic', CS)):
        v = d.loc[d.grp == grp, 'theta_sd']
        ax.hist(v, bins=12, alpha=.55, color=col, label=grp, density=True)
    ax.set_xlabel(r'$\theta$ SD (deg)'); ax.set_ylabel('density')
    ax.legend(frameon=False)
    ax.set_title('(b) bending amplitude', loc='left')

    ax = axes[1, 1]
    g = d.groupby('fish')['asym_dur'].agg(['mean', 'sem'])
    g['sc'] = g.index.str.startswith('sf')
    g = g.sort_values('mean')
    ax.errorbar(range(len(g)), g['mean'], yerr=g['sem'], fmt='none',
                ecolor='0.4', lw=.8, capsize=2)
    ax.scatter(range(len(g)), g['mean'], s=26,
               c=[CS if v else CH for v in g.sc], ec='k', lw=.4, zorder=3)
    ax.axhline(0, color='k', lw=.6, ls='--')
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=45,
                                                     fontsize=6.5)
    ax.set_ylabel('half-cycle duration\nasymmetry')
    ax.set_title('(d) left-right asymmetry (mean $\\pm$ SEM)', loc='left')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print('ok', fname)


# ---------------------------------------------------------------- fig 5
def fig_pseudoreplication(fname='fig_pseudoreplication.pdf'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    dd = pd.read_csv('features_all11.csv')
    dd['y'] = dd.fish.str.startswith('sf').astype(int)
    cols = ['SampEn', 'distance_cm']
    base = 100 * (1 - dd.y.mean())

    sub = dd[dd.fish.isin(['f2', 'f3', 'sf2', 'sf3'])]
    m = make_pipeline(StandardScaler(), LDA()).fit(sub[cols], sub.y)
    a_sub = m.score(sub[cols], sub.y) * 100
    m = make_pipeline(StandardScaler(), LDA()).fit(dd[cols], dd.y)
    a_all = m.score(dd[cols], dd.y) * 100
    yt, yp = [], []
    for f_ in dd.fish.unique():
        tr, te = dd[dd.fish != f_], dd[dd.fish == f_]
        mm = make_pipeline(StandardScaler(), LDA()).fit(tr[cols], tr.y)
        yp.extend(mm.predict(te[cols])); yt.extend(te.y)
    yt, yp = np.array(yt), np.array(yp)
    a_cv = (yt == yp).mean() * 100
    sens = yp[yt == 1].mean() * 100

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    lab = ['2 vs 2 subset,\nin-sample', 'all 10 fish,\nin-sample',
           'leave-one-fish-out\ncross-validation']
    val = [a_sub, a_all, a_cv]
    b = ax.bar(range(3), val, color=['#d1495b', '#e8a33d', '#2c6fb5'],
               ec='k', lw=.6, width=.62)
    ax.axhline(base, color='k', ls='--', lw=.8)
    ax.text(2.46, base + 1.5, 'majority-class\nbaseline', ha='right', fontsize=6.2)
    for i, (r, v) in enumerate(zip(b, val)):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f'{v:.1f}%',
                ha='center', fontsize=7.5, fontweight='bold')
    ax.text(2, 40, f'sensitivity\n{sens:.0f}%', ha='center', fontsize=7.5,
            color='white', fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=6.6)
    ax.set_ylabel('classification accuracy (%)')
    ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print('ok', fname, dict(sub=a_sub, all=a_all, cv=a_cv, sens=sens))
    json.dump(dict(acc_subset=a_sub, acc_all=a_all, acc_cv=a_cv,
                   sens_cv=sens), open('clf_numbers.json', 'w'))


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fns = dict(hc=fig_complexity_plane, sur=fig_surrogates,
               att=fig_no_attractor, sig=fig_signal,
               pse=fig_pseudoreplication)
    for k, fn in fns.items():
        if which in ('all', k):
            fn()
