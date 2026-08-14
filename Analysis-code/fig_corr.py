"""
Redundancy structure of the feature set.

(a) replicates the correlation matrix of the seven features used in the
    previous analysis of this dataset, on the eleven animals of the present
    cohort. Distance travelled and mean speed are algebraically the same
    quantity here (the recording window is fixed at 30 s), so their
    correlation of 1.00 is an identity rather than a measurement.

(b) full correlation matrix of the thirty-three features, ordered by class.
    Estimated at trial level because a 33x33 correlation matrix is rank
    deficient with eleven observations.

The effective number of independent features is obtained from the eigenvalue
spectrum following Li & Ji (2005), with the Cheverud-Nyholt estimator given
for comparison.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.6, 'savefig.bbox': 'tight',
                     'figure.dpi': 300})
CH, CS = '#2c6fb5', '#d1495b'

CLASSES = [
    ('kinematic', ['distance_cm', 'mean_speed', 'max_speed', 'max_accel',
                   'theta_sd', 'beat_rate']),
    ('geometric', ['hull_area', 'Rg', 'occupancy', 'd_wall', 'spread_index']),
    ('symmetry', ['asym_dur', 'asym_count', 'asym_amp', 'theta_mean',
                  'theta_median', 'theta_skew', 'theta_kurt']),
    ('entropy &\ninformation', ['SampEn', 'ApEn', 'PE_H5', 'PE_C5',
                                'PE_missing5', 'CI', 'AIS']),
    ('recurrence &\nscaling', ['DET', 'ENTR', 'TT', 'RN_trans', 'DFA_alpha',
                               'alpha_width']),
    ('attractor', ['D2', 'lambda1']),
]
NICE = {
    'distance_cm': 'distance', 'mean_speed': 'mean speed',
    'max_speed': 'max speed', 'max_accel': 'max acceleration',
    'theta_sd': r'$\theta$ SD', 'beat_rate': 'beat rate',
    'hull_area': 'hull area', 'Rg': r'$R_g$', 'occupancy': 'occupancy',
    'd_wall': 'path-boundary distance', 'spread_index': 'spread index',
    'asym_dur': 'asym. duration', 'asym_count': 'asym. count',
    'asym_amp': 'asym. amplitude', 'theta_mean': r'$\theta$ mean',
    'theta_median': r'$\theta$ median', 'theta_skew': r'$\theta$ skew',
    'theta_kurt': r'$\theta$ kurtosis',
    'SampEn': 'SampEn', 'ApEn': 'ApEn', 'PE_H5': 'perm. entropy',
    'PE_C5': 'stat. complexity', 'PE_missing5': 'forbidden patterns',
    'CI': 'multiscale index', 'AIS': 'info. storage',
    'DET': 'DET', 'ENTR': 'ENTR', 'TT': 'TT', 'RN_trans': 'RN transitivity',
    'DFA_alpha': r'DFA $\alpha$', 'alpha_width': 'MFDFA width',
    'D2': r'$D_2$', 'lambda1': r'$\lambda_1$',
}
FEATURES = [f for _, fs in CLASSES for f in fs]

d = pd.read_csv('features_final.csv')
d['y'] = d.fish.str.startswith('sf').astype(int)
g = d.groupby('fish')[FEATURES].mean()
gy = d.groupby('fish').y.first().loc[g.index].values


def n_eff(C):
    """Effective number of independent variables from the eigenvalue spectrum."""
    lam = np.clip(np.sort(np.linalg.eigvalsh(C))[::-1], 0, None)
    li_ji = float(np.sum((lam >= 1) + (lam - np.floor(lam))))
    M = C.shape[0]
    chev = float(1 + (M - 1) * (1 - np.var(lam, ddof=1) / M))
    return li_ji, chev, lam


# ------------------------------------------------------------------ figure
FW, FH = 6.30, 8.80
fig = plt.figure(figsize=(FW, FH))


def inch(x_in, y_in, w_in, h_in):
    """Axes rectangle given in inches from the bottom-left of the figure."""
    return [x_in / FW, y_in / FH, w_in / FW, h_in / FH]


def X(x_in):
    return x_in / FW


def Y(y_in):
    return y_in / FH


cmap = plt.get_cmap('RdBu_r')
MAT_L = 1.35                              # left edge shared by both matrices
CB_W, CB_GAP = 0.15, 0.20


def colourbar(mappable, left, bottom, side, frac=0.62):
    """Vertical colour key beside a square matrix, centred on it."""
    h = side * frac
    cax = fig.add_axes(inch(left + side + CB_GAP, bottom + (side - h) / 2,
                            CB_W, h))
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label('Pearson $r$', fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8, length=1.8, pad=1.6)
    cb.outline.set_linewidth(0.4)
    return cb


# ---------- (a) the seven previously used features, on our eleven animals
ORIG = ['distance_cm', 'mean_speed', 'max_speed', 'max_accel', 'ApEn',
        'D2', 'lambda1']
SHORT = ['DT', 'speed', 'MaxV', 'MaxA', 'ApEn', r'$D_2$', r'$\lambda_1$']
Ca = g[ORIG].corr().values
n = len(ORIG)

A_TOP, A_SIDE = 8.55, 3.25
A_B = A_TOP - A_SIDE
cell = A_SIDE / n
pad = 0.016

for i in range(n):
    for j in range(n):
        ax = fig.add_axes(inch(MAT_L + j * cell + pad,
                               A_TOP - (i + 1) * cell + pad,
                               cell - 2 * pad, cell - 2 * pad))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('0.55')
        r = Ca[i, j]
        if i == j:                        # diagonal: distribution over animals
            v = g[ORIG[i]].values
            ax.hist([v[gy == 0], v[gy == 1]], bins=5, stacked=True,
                    color=[CH, CS], ec='k', lw=0.3)
            ax.margins(x=0.06)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        elif i < j:                       # upper triangle: coefficients
            ax.set_facecolor(cmap(0.5 + r / 2))
            ax.text(.5, .5, f'{r:.2f}', ha='center', va='center', fontsize=10,
                    transform=ax.transAxes,
                    color='white' if abs(r) > .55 else '0.1',
                    fontweight='bold' if abs(r) > .75 else 'normal')
        else:                             # lower triangle: scatter and fit
            x, y = g[ORIG[j]].values, g[ORIG[i]].values
            xs = (x - x.mean()) / x.std(); ys = (y - y.mean()) / y.std()
            for c, col, mk in ((0, CH, 'o'), (1, CS, '^')):
                m = gy == c
                ax.scatter(xs[m], ys[m], s=7, c=col, marker=mk, lw=0.2,
                           ec='k', alpha=.9, zorder=3)
            b = np.polyfit(xs, ys, 1)
            xx = np.array([xs.min(), xs.max()])
            ax.plot(xx, np.polyval(b, xx), color='#7d3c98', lw=0.9, zorder=2)
            ax.margins(0.18)
        if (i, j) in ((0, 1), (1, 0)):    # the algebraic identity
            for sp in ax.spines.values():
                sp.set_linewidth(1.8); sp.set_color('#1a7a3c')
        if j == 0:
            fig.text(X(MAT_L - 0.07), Y(A_TOP - (i + 0.5) * cell), SHORT[i],
                     fontsize=8.4, ha='right', va='center')
        if i == n - 1:
            fig.text(X(MAT_L + (j + 0.5) * cell), Y(A_B - 0.055), SHORT[j],
                     fontsize=8.4, ha='center', va='top')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
colourbar(sm, MAT_L, A_B, A_SIDE)

fig.text(X(MAT_L), Y(A_TOP + 0.07),
         '(a) the seven features of the previous analysis',
         fontsize=8.5, va='bottom')

fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc=CH, mec='k',
                           mew=.3, ms=5.5, label='healthy'),
                    Line2D([], [], marker='^', ls='', mfc=CS, mec='k',
                           mew=.3, ms=5.5, label='scoliotic')],
           loc='center', bbox_to_anchor=(X(MAT_L + A_SIDE / 2), Y(4.94)),
           frameon=False, fontsize=8.4, ncol=2, handletextpad=.35,
           columnspacing=1.6)

# ---------- (b) the full feature set
Cb = d[FEATURES].corr().values
liji, chev, lam = n_eff(Cb)

B_SIDE, B_B = 4.05, 0.23
axb = fig.add_axes(inch(MAT_L, B_B, B_SIDE, B_SIDE))
im = axb.imshow(Cb, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest',
                aspect='auto')

edges, acc = [], 0
for _, fs in CLASSES:
    acc += len(fs); edges.append(acc)
for e in edges[:-1]:
    axb.axhline(e - .5, color='k', lw=0.9)
    axb.axvline(e - .5, color='k', lw=0.9)

axb.set_xticks([])
axb.set_yticks(range(len(FEATURES)))
axb.set_yticklabels([NICE[f] for f in FEATURES], fontsize=6.8)
axb.tick_params(length=1.6, pad=1.4)

start = 0
for (nm, fs), e in zip(CLASSES, edges):
    mid = (start + e - 1) / 2
    axb.text(mid, -1.15, nm, fontsize=6.6, ha='center', va='bottom',
             color='0.2', linespacing=1.15)
    axb.plot([start - .4, e - .6], [-0.80, -0.80], color='0.45', lw=0.8,
             clip_on=False)
    start = e
axb.set_ylim(len(FEATURES) - .5, -.5)

k = FEATURES.index('asym_dur')
axb.add_patch(plt.Rectangle((-.5, k - .5), len(FEATURES), 1, fill=False,
                            ec='#1a7a3c', lw=1.3, zorder=4))

colourbar(im, MAT_L, B_B, B_SIDE)

fig.text(X(MAT_L), Y(4.66),
         '(b) all thirty-three features, grouped by class; effective number '
         'of independent features: %.0f of 33' % liji,
         fontsize=8.5, va='bottom')

fig.savefig('fig_corr.pdf')
plt.close(fig)

# ------------------------------------------------------------------ numbers
res = dict(
    dt_speed_animal=float(g.distance_cm.corr(g.mean_speed)),
    dt_speed_trial=float(d.distance_cm.corr(d.mean_speed)),
    dt_apen_animal=float(g.distance_cm.corr(g.ApEn)),
    dt_apen_trial=float(d.distance_cm.corr(d.ApEn)),
    dt_maxv_animal=float(g.distance_cm.corr(g.max_speed)),
    n_eff_liji=liji, n_eff_chev=chev,
    lam1=float(lam[0]), lam1_pct=float(100 * lam[0] / len(FEATURES)),
)
# correlation of asym_dur with everything else
a = np.abs(pd.Series(Cb[k], index=FEATURES).drop('asym_dur'))
res['asym_dur_max_abs_r'] = float(a.max())
res['asym_dur_max_partner'] = str(a.idxmax())
res['asym_dur_median_abs_r'] = float(a.median())
json.dump(res, open('corr_numbers.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
