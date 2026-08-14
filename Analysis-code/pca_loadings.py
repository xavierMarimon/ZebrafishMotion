"""
Composition of the principal components of the thirty-three-feature matrix,
with a bootstrap estimate of how precisely each coefficient is determined.

A table of component coefficients invites the reader to interpret what each
component "is". With eleven animals that interpretation is fragile, so every
coefficient is reported with the standard deviation of its bootstrap
distribution over animals.

Two technical points about the bootstrap. The sign of an eigenvector is
arbitrary, so each resampled component is aligned to the observed one by the
sign of their inner product. The *order* of components is also not
guaranteed: when two eigenvalues are close, resampling can exchange them.
Each resampled component is therefore matched to the observed component with
which it has the largest absolute inner product (greedy, without
replacement), and the proportion of resamples in which the match preserves
the original rank is reported as a measure of how well identified the
component itself is.

Also reproduces, on the present cohort, the seven-feature analysis of the
previous work, whose last component is the algebraic signature of a pair of
perfectly collinear features.

Outputs
-------
pca_loadings.csv   coefficients and bootstrap SDs for every feature and PC
pca_numbers.json   scalars quoted in the manuscript
tab_pca.tex        the LaTeX table
"""
import json
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

N_BOOT = 5000
N_COMP = 5        # components tabulated
TOP = 6           # "leading contributor" = among the TOP largest |coefficients|
BOLD = 0.25       # coefficients at least this large are set in bold
SEED = 0

CLASSES = [
    ('Kinematic', ['distance_cm', 'mean_speed', 'max_speed', 'max_accel',
                   'theta_sd', 'beat_rate']),
    ('Geometric', ['hull_area', 'Rg', 'occupancy', 'd_wall', 'spread_index']),
    ('Symmetry', ['asym_dur', 'asym_count', 'asym_amp', 'theta_mean',
                  'theta_median', 'theta_skew', 'theta_kurt']),
    ('Entropy and information', ['SampEn', 'ApEn', 'PE_H5', 'PE_C5',
                                 'PE_missing5', 'CI', 'AIS']),
    ('Recurrence and scaling', ['DET', 'ENTR', 'TT', 'RN_trans', 'DFA_alpha',
                                'alpha_width']),
    ('Attractor invariants', ['D2', 'lambda1']),
]
FEATURES = [f for _, fs in CLASSES for f in fs]
NICE = {
    'distance_cm': r'Distance travelled, $d_T$ (cm)',
    'mean_speed': r'Mean speed, $\bar{v}$ (cm\,s$^{-1}$)',
    'max_speed': r'Maximum speed, $v_{\max}$ (cm\,s$^{-1}$)',
    'max_accel': r'Maximum acceleration, $a_{\max}$ (cm\,s$^{-2}$)',
    'theta_sd': r'Bending amplitude, $\sigma_{\theta}$ (deg)',
    'beat_rate': r'Tail-beat rate, $f_{\mathrm{beat}}$ (Hz)',
    'hull_area': r'Convex-hull area, $A_{\mathrm{hull}}$ (cm$^{2}$)',
    'Rg': r'Radius of gyration, $R_g$ (cm)',
    'occupancy': r'Grid occupancy, $N_{\mathrm{cells}}$ (cells)',
    'd_wall': r'Mean distance to path boundary, $\bar{d}_{\mathrm{edge}}$ (cm)',
    'spread_index': r'Spread index, $A_{\mathrm{hull}}/d_T^{2}$ (--)',
    'asym_dur': r'Half-cycle duration asymmetry, $\mathcal{A}_{\tau}$ (--)',
    'asym_count': r'Half-cycle count asymmetry, $\mathcal{A}_{N}$ (--)',
    'asym_amp': r'Half-cycle amplitude asymmetry, $\mathcal{A}_{\theta}$ (--)',
    'theta_mean': r'Mean bending angle, $\bar{\theta}$ (deg)',
    'theta_median': r'Median bending angle, $\tilde{\theta}$ (deg)',
    'theta_skew': r'Bending angle skewness, $g_1$ (--)',
    'theta_kurt': r'Bending angle excess kurtosis, $g_2$ (--)',
    'SampEn': r'Sample entropy, SampEn (nats)',
    'ApEn': r'Approximate entropy, ApEn (nats)',
    'PE_H5': r'Permutation entropy, $H$ (--)',
    'PE_C5': r'Statistical complexity, $C_{JS}$ (--)',
    'PE_missing5': r'Forbidden ordinal patterns, $f_{\mathrm{forb}}$ (--)',
    'CI': r'MSE complexity index, CI (--)',
    'AIS': r'Active information storage, AIS (nats)',
    'DET': r'Determinism, $DET$ (--)',
    'ENTR': r'Recurrence entropy, $ENTR$ (nats)',
    'TT': r'Trapping time, $TT$ (samples)',
    'RN_trans': r'Recurrence-network transitivity, $\mathcal{T}$ (--)',
    'DFA_alpha': r'DFA exponent, $\alpha$ (--)',
    'alpha_width': r'Multifractal width, $\Delta\alpha$ (--)',
    'D2': r'Correlation dimension, $D_2$ (--)',
    'lambda1': r'Largest Lyapunov exponent, $\lambda_1$ (s$^{-1}$)',
}


def pca(A):
    """Correlation-matrix PCA. Returns eigenvalues and unit-norm eigenvectors."""
    Z = (A - A.mean(0)) / A.std(0, ddof=1)
    lam, V = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    o = np.argsort(lam)[::-1]
    return np.clip(lam[o], 0, None), V[:, o]


def orient(V):
    """Fix the arbitrary sign so that the largest contributor is positive."""
    for c in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, c])), c] < 0:
            V[:, c] *= -1
    return V


def bootstrap(A, V, n_boot=N_BOOT, n_comp=N_COMP, seed=SEED):
    """Bootstrap over animals. Returns coefficient SDs, rank-preservation
    rates, and the probability of remaining among the TOP contributors."""
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_boot, A.shape[1], n_comp))
    same_rank = np.zeros(n_comp)
    in_top = np.zeros((A.shape[1], n_comp))
    used = 0
    for _ in range(n_boot):
        Ab = A[rng.integers(0, len(A), len(A))]
        if np.any(Ab.std(0, ddof=1) == 0):
            continue
        _, Vb = pca(Ab)
        taken = set()
        for c in range(n_comp):
            dots = np.abs(Vb.T @ V[:, c])
            dots[list(taken)] = -1.0
            j = int(np.argmax(dots))
            taken.add(j)
            same_rank[c] += (j == c)
            v = Vb[:, j] * np.sign(np.dot(Vb[:, j], V[:, c]))
            draws[used, :, c] = v
            for t in np.argsort(np.abs(v))[::-1][:TOP]:
                in_top[t, c] += 1
        used += 1
    return (draws[:used].std(0, ddof=1), same_rank / used, in_top / used, used)


# --------------------------------------------------------------- LaTeX table
def make_table(V, sd, var, rank, n_boot, equal_w):
    """Features as rows, components as columns; under each coefficient, the
    standard deviation of its bootstrap distribution."""
    head = [
        r'%---------------------------------------------------------------- TAB PCA',
        r'\begin{table}[!ht]',
        r'\caption{Composition of the first five principal components of the '
        r'thirty-three-feature matrix, which together account for '
        r'$%.1f\%%$ of the variance. Each entry is the coefficient of the '
        r'feature in the unit-norm eigenvector; the smaller figure beneath it '
        r'is the standard deviation of its bootstrap distribution over the '
        r'eleven animals ($%d$ resamples). A feature contributing equally to a '
        r'component would carry $1/\sqrt{33}=%.3f$; coefficients of at least '
        r'$%.2f$ in absolute value are set in bold. The sign of a component is '
        r'arbitrary and is fixed here so that its largest contributor is '
        r'positive. Bootstrap components are matched to the observed ones by '
        r'inner product rather than by rank, because resampling can exchange '
        r'components with similar eigenvalues; the last row gives the '
        r'proportion of resamples in which the match preserved the original '
        r'rank.}'
        % (var[:N_COMP].sum(), n_boot, equal_w, BOLD),
        r'\label{tab:pca}',
        r'\centering',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{0pt}',
        r'\renewcommand{\arraystretch}{0.95}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l'
        + 'r' * N_COMP + r'@{}}',
        r'\toprule',
        r'& \multicolumn{%d}{c}{Principal components}\\' % N_COMP,
        r'\cmidrule(l){2-%d}' % (N_COMP + 1),
        'Features & ' + ' & '.join(r'PC%d' % (c + 1) for c in range(N_COMP))
        + r'\\',
        '(variance explained) & '
        + ' & '.join(r'(%.1f\%%)' % var[c] for c in range(N_COMP)) + r'\\',
        r'\midrule',
    ]
    body = []
    for cls, feats in CLASSES:
        body.append(r'\multicolumn{%d}{@{}l}{\itshape %s}\\[1pt]'
                    % (N_COMP + 1, cls))
        for f in feats:
            i = FEATURES.index(f)
            coef, dev = [], []
            for c in range(N_COMP):
                v = V[i, c]
                coef.append(r'\textbf{%+.4f}' % v if abs(v) >= BOLD
                            else '%+.4f' % v)
                dev.append(r'{\tiny\color{black!55} $\pm$%.3f}' % sd[i, c])
            body.append(r'\quad %s & ' % NICE[f] + ' & '.join(coef)
                        + r'\\[-1.0pt]')
            body.append(r'& ' + ' & '.join(dev) + r'\\[2.0pt]')
        body.append(r'\addlinespace[1pt]')
    body = body[:-1]
    foot = [
        r'\midrule',
        r'Rank preserved under resampling & '
        + ' & '.join(r'%.0f\%%' % (100 * rank[c]) for c in range(N_COMP))
        + r'\\',
        r'\bottomrule',
        r'\end{tabular*}',
        r'\end{table}',
    ]
    return '\n'.join(head + body + foot)


def main():
    d = pd.read_csv('features_final.csv')
    g = d.groupby('fish')[FEATURES].mean()
    A = g.values
    lam, V = pca(A)
    V = orient(V)
    var = 100 * lam / lam.sum()
    sd, rank, in_top, used = bootstrap(A, V)
    equal_w = 1 / np.sqrt(len(FEATURES))

    cols = {}
    for c in range(N_COMP):
        cols['PC%d' % (c + 1)] = V[:, c]
        cols['sd%d' % (c + 1)] = sd[:, c]
        cols['top%d' % (c + 1)] = in_top[:, c]
    pd.DataFrame(cols, index=FEATURES).to_csv('pca_loadings.csv')
    open('tab_pca.tex', 'w').write(
        make_table(V, sd, var, rank, used, equal_w))

    # ---- the seven features of the previous analysis ---------------------
    ORIG = ['distance_cm', 'mean_speed', 'max_speed', 'max_accel', 'ApEn',
            'D2', 'lambda1']
    lam7, V7 = pca(g[ORIG].values)
    last = V7[:, -1]
    if last[np.argmax(np.abs(last))] < 0:
        last = -last

    out = dict(
        var=[float(x) for x in var[:N_COMP]],
        cum=[float(x) for x in np.cumsum(var[:N_COMP])],
        rank_preserved=[float(x) for x in rank],
        sd_median=[float(x) for x in np.median(sd, 0)],
        n_animals=int(len(A)), n_features=len(FEATURES), n_boot=int(used),
        equal_weight=float(equal_w),
        max_pc1=float(np.abs(V[:, 0]).max()),
        apen_pc1=float(V[FEATURES.index('ApEn'), 0]),
        apen_pc1_sd=float(sd[FEATURES.index('ApEn'), 0]),
        dist_pc1=float(V[FEATURES.index('distance_cm'), 0]),
        dist_pc1_sd=float(sd[FEATURES.index('distance_cm'), 0]),
        dist_top1=float(in_top[FEATURES.index('distance_cm'), 0]),
        speed_top1=float(in_top[FEATURES.index('mean_speed'), 0]),
        apen_top1=float(in_top[FEATURES.index('ApEn'), 0]),
        lam7_last=float(lam7[-1]),
        lam7_last_pct=float(100 * lam7[-1] / 7),
        v7_last={k: float(v) for k, v in zip(ORIG, last)},
    )
    json.dump(out, open('pca_numbers.json', 'w'), indent=1)

    print('PCA on %d animals, %d features (%d usable resamples)'
          % (out['n_animals'], out['n_features'], used))
    print('variance %:', np.round(var[:N_COMP], 1),
          ' cumulative:', np.round(out['cum'], 1))
    print('rank preserved %:', np.round(100 * rank, 1))
    print('median bootstrap SD per component:', np.round(out['sd_median'], 3))
    print('equal contribution 1/sqrt(33) = %.3f' % equal_w)
    print('ApEn on PC1  %+.4f +- %.3f  (top-6 in %.0f%% of resamples)'
          % (out['apen_pc1'], out['apen_pc1_sd'], 100 * out['apen_top1']))
    print('distance PC1 %+.4f +- %.3f  (top-6 in %.0f%% of resamples)'
          % (out['dist_pc1'], out['dist_pc1_sd'], 100 * out['dist_top1']))
    print('seven-feature PCA: smallest eigenvalue %.6f (%.4f%% of variance)'
          % (out['lam7_last'], out['lam7_last_pct']))
    print('  its eigenvector:', {k: round(v, 3)
                                 for k, v in out['v7_last'].items()})


if __name__ == '__main__':
    main()
