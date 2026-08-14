"""
Supervised classification in four specified feature spaces.

No feature selection takes place here: each space is fixed in advance, either
because the previous analysis of this dataset used it or because the feature
was derived from the mechanics of a lateral spinal curvature before the data
were examined. That is what makes the permutation test in this section exact
and interpretable, in contrast to the automatic-selection procedure of
Sect. "Unsupervised structure", where the choice of how many features to keep
must itself be included in the null.

Protocol for every panel:
  * the unit of analysis is the animal, never the trial;
  * a linear discriminant is fitted to the eleven animal means and evaluated
    by leave-one-animal-out cross-validation;
  * the null distribution is obtained by enumerating all C(11,3) = 165 ways
    of assigning three animals to the pathological group, which makes the
    p-value exact. Its smallest attainable value is 1/165 = 0.0061.

Outputs fig_lda.pdf and lda_numbers.json.
"""
import json
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.7, 'savefig.bbox': 'tight',
                     'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'figure.dpi': 300})
CH, CS = '#2c6fb5', '#d1495b'
FIELD = ListedColormap(['#dce9f5', '#f8dfe3'])      # decision regions

FEATURES = ['distance_cm', 'mean_speed', 'max_speed', 'max_accel', 'theta_sd',
            'beat_rate', 'hull_area', 'Rg', 'occupancy', 'd_wall',
            'spread_index', 'asym_dur', 'asym_count', 'asym_amp',
            'theta_mean', 'theta_median', 'theta_skew', 'theta_kurt',
            'SampEn', 'ApEn', 'PE_H5', 'PE_C5', 'PE_missing5', 'CI', 'AIS',
            'DET', 'ENTR', 'TT', 'RN_trans', 'DFA_alpha', 'alpha_width',
            'D2', 'lambda1']


def clf():
    return make_pipeline(StandardScaler(), LDA())


def lofo(X, y):
    """Leave-one-animal-out prediction. Nothing is fitted on the held-out animal."""
    pred = np.zeros(len(y), int)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        if len(np.unique(y[m])) < 2:
            continue
        pred[i] = clf().fit(X[m], y[m]).predict(X[i:i + 1])[0]
    sens = pred[y == 1].mean()
    spec = 1 - pred[y == 0].mean()
    return pred, dict(acc=float((pred == y).mean()), sens=float(sens),
                      spec=float(spec), bal=float((sens + spec) / 2))


def exact_p(X, y):
    """Exact permutation test: enumerate every assignment of three animals."""
    obs = lofo(X, y)[1]['bal']
    n, k = len(y), int(y.sum())
    ge = tot = 0
    for c in combinations(range(n), k):
        z = np.zeros(n, int)
        z[list(c)] = 1
        tot += 1
        ge += lofo(X, z)[1]['bal'] >= obs - 1e-12
    return float(ge / tot), int(tot)


def main():
    d = pd.read_csv('features_final.csv')
    d['y'] = d.fish.str.startswith('sf').astype(int)
    g = d.groupby('fish')[FEATURES].mean()
    fish = g.index.to_numpy()
    y = d.groupby('fish').y.first().loc[g.index].values

    # principal components of the standardised animal-level matrix
    sc = StandardScaler().fit(g[FEATURES])
    pca = PCA().fit(sc.transform(g[FEATURES]))
    S = pca.transform(sc.transform(g[FEATURES]))
    St = pca.transform(sc.transform(d[FEATURES]))     # trials in the same space

    panels = [
        dict(key='prev', cols=['distance_cm', 'SampEn'],
             X=g[['distance_cm', 'SampEn']].values,
             Xt=d[['distance_cm', 'SampEn']].values,
             xlab='distance travelled (cm)', ylab='sample entropy',
             title='features of the previous analysis'),
        dict(key='pc', cols=None, X=S[:, :2], Xt=St[:, :2],
             xlab='PC1 (%.1f%% of variance)'
                  % (100 * pca.explained_variance_ratio_[0]),
             ylab='PC2 (%.1f%%)'
                  % (100 * pca.explained_variance_ratio_[1]),
             title='first two principal components'),
        dict(key='sym', cols=['asym_dur', 'asym_count'],
             X=g[['asym_dur', 'asym_count']].values,
             Xt=d[['asym_dur', 'asym_count']].values,
             xlab='half-cycle duration asymmetry',
             ylab='half-cycle count asymmetry',
             title='pre-specified symmetry pair'),
        dict(key='dur', cols=['asym_dur'], X=g[['asym_dur']].values,
             Xt=d[['asym_dur']].values,
             xlab='half-cycle duration asymmetry', ylab=None,
             title='half-cycle duration asymmetry alone'),
    ]
    for p in panels:
        p['pred'], p['m'] = lofo(p['X'], y)
        p['p'], p['ntot'] = exact_p(p['X'], y)

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.0))
    letters = 'abcd'
    for k, (ax, p) in enumerate(zip(axes.ravel(), panels)):
        X, Xt, pred = p['X'], p['Xt'], p['pred']
        if X.shape[1] == 2:
            model = clf().fit(X, y)
            lo, hi = X.min(0), X.max(0)
            pad = 0.28 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 400),
                                 np.linspace(lo[1], hi[1], 400))
            zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.pcolormesh(xx, yy, zz, cmap=FIELD, shading='auto', zorder=0)
            ax.contour(xx, yy, zz, levels=[0.5], colors='0.35',
                       linewidths=0.9, zorder=1)
            for c, col, mk in ((0, CH, 'o'), (1, CS, '^')):
                m = d.y.values == c
                ax.scatter(Xt[m, 0], Xt[m, 1], s=5, c=col, marker=mk,
                           alpha=0.22, lw=0, zorder=2, clip_on=True)
            for c, col, mk in ((0, CH, 'o'), (1, CS, '^')):
                m = y == c
                ax.scatter(X[m, 0], X[m, 1], s=52, c=col, marker=mk,
                           ec='k', lw=0.6, zorder=4)
            wrong = pred != y
            if wrong.any():
                ax.scatter(X[wrong, 0], X[wrong, 1], s=190, facecolors='none',
                           ec='k', lw=1.2, marker='o', zorder=5)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_ylabel(p['ylab'])
        else:                                   # one-dimensional panel
            model = clf().fit(X, y)
            lo, hi = float(X.min()), float(X.max())
            pad = 0.45 * (hi - lo); lo, hi = lo - pad, hi + pad
            grid = np.linspace(lo, hi, 800)[:, None]
            zz = model.predict(grid)
            ax.pcolormesh(grid.ravel(), np.array([0.0, 1.0]),
                          np.vstack([zz, zz]), cmap=FIELD,
                          shading='nearest', zorder=0)
            cut = grid.ravel()[np.argmax(np.diff(zz) != 0) + 1]
            ax.axvline(cut, color='0.35', lw=0.9, zorder=1)
            rng = np.random.default_rng(1)
            ROW = {0: 0.30, 1: 0.70}
            for c, col, mk in ((0, CH, 'o'), (1, CS, '^')):
                m = d.y.values == c
                ax.scatter(Xt[m, 0],
                           ROW[c] + rng.uniform(-0.13, 0.13, m.sum()), s=5,
                           c=col, marker=mk, alpha=0.22, lw=0, zorder=2)
            # small alternating offsets so animals with similar values
            # do not sit on top of one another
            yy_ = np.zeros(len(y))
            for c in (0, 1):
                idx = np.where(y == c)[0]
                order = idx[np.argsort(X[idx, 0])]
                for r, i in enumerate(order):
                    yy_[i] = ROW[c] + (0.055 if r % 2 else -0.055)
            for c, col, mk in ((0, CH, 'o'), (1, CS, '^')):
                m = y == c
                ax.scatter(X[m, 0], yy_[m], s=52, c=col, marker=mk, ec='k',
                           lw=0.6, zorder=4)
            wrong = pred != y
            if wrong.any():
                ax.scatter(X[wrong, 0], yy_[wrong], s=190, facecolors='none',
                           ec='k', lw=1.2, marker='o', zorder=5)
            ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
            ax.set_yticks([0.30, 0.70])
            ax.set_yticklabels(['healthy', 'scoliotic'], rotation=90,
                               va='center')
        ax.set_xlabel(p['xlab'])
        ax.set_title('(%s) %s\nbalanced accuracy %.2f, exact $p=%.3f$'
                     % (letters[k], p['title'], p['m']['bal'], p['p']),
                     loc='left')

    fig.legend(handles=[
        Line2D([], [], marker='o', ls='', mfc=CH, mec='k', mew=.4, ms=6,
               label='healthy animal'),
        Line2D([], [], marker='^', ls='', mfc=CS, mec='k', mew=.4, ms=6,
               label='scoliotic animal'),
        Line2D([], [], marker='o', ls='', mfc='none', mec='k', mew=1.1, ms=9,
               label='misclassified when held out'),
        Line2D([], [], marker='o', ls='', mfc='0.6', mec='none', ms=3,
               label='individual trials')],
        loc='lower center', bbox_to_anchor=(0.5, -0.055), frameon=False,
        ncol=4, handletextpad=.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    fig.savefig('fig_lda.pdf')
    plt.close(fig)

    out = {p['key']: dict(p['m'], p=p['p'], n_assignments=p['ntot'],
                          misclassified=[str(f) for f in
                                         fish[p['pred'] != y]])
           for p in panels}
    out['floor_p'] = 1 / panels[0]['ntot']
    json.dump(out, open('lda_numbers.json', 'w'), indent=1)
    for p in panels:
        m = p['m']
        print('%-34s acc %.3f  sens %.2f  spec %.2f  bal %.3f  p=%.4f  '
              'missed: %s' % (p['title'], m['acc'], m['sens'], m['spec'],
                              m['bal'], p['p'],
                              ', '.join(fish[p['pred'] != y]) or 'none'))
    print('smallest attainable p = 1/%d = %.4f' % (panels[0]['ntot'],
                                                   out['floor_p']))


if __name__ == '__main__':
    main()
