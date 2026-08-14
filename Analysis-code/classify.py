"""
Unsupervised and supervised analysis of the feature matrix.

Design points that differ from the original pipeline:

* Feature selection is performed *inside* the cross-validation loop. Selecting
  features on the full dataset and then cross-validating the classifier leaks
  the labels of the held-out animal into the model and inflates performance;
  this is a documented source of optimistic bias (Ambroise & McLachlan 2002).
* The held-out unit is the animal, never the trial, because trials from one
  animal are not independent.
* Classification accuracy is itself tested against a null distribution
  obtained by permuting the group labels *at the animal level*, which is the
  only exchangeable unit here (Ojala & Garriga 2010).
"""
import json
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy import stats

FEATURES = [
    # kinematic
    'distance_cm', 'mean_speed', 'max_speed', 'max_accel', 'theta_sd',
    'beat_rate',
    # geometric
    'hull_area', 'Rg', 'occupancy', 'd_wall', 'spread_index',
    # symmetry
    'asym_dur', 'asym_count', 'asym_amp', 'theta_mean', 'theta_median',
    'theta_skew', 'theta_kurt',
    # entropy & information
    'SampEn', 'ApEn', 'PE_H5', 'PE_C5', 'PE_missing5', 'CI', 'AIS',
    # recurrence & scaling
    'DET', 'ENTR', 'TT', 'RN_trans', 'DFA_alpha', 'alpha_width',
    # attractor invariants
    'D2', 'lambda1',
]


def load(level='animal'):
    d = pd.read_csv('features_final.csv')
    d['y'] = d.fish.str.startswith('sf').astype(int)
    if level == 'animal':
        g = d.groupby('fish')[FEATURES].mean()
        y = d.groupby('fish').y.first().loc[g.index].values
        return g.index.to_numpy(), g.values, y
    return d.fish.to_numpy(), d[FEATURES].values, d.y.values


# ----------------------------------------------------------------- selection
def select_features(X, y, k):
    """Rank features by absolute standardised mean difference on X, y only."""
    a, b = X[y == 1], X[y == 0]
    sd = np.sqrt(((a.shape[0] - 1) * a.var(0, ddof=1) +
                  (b.shape[0] - 1) * b.var(0, ddof=1)) /
                 max(1, a.shape[0] + b.shape[0] - 2))
    sd[sd == 0] = np.inf
    score = np.abs(a.mean(0) - b.mean(0)) / sd
    return np.argsort(score)[::-1][:k]


def lofo_predict(groups, X, y, k):
    """
    Leave-one-animal-out prediction with feature selection nested inside the
    loop. Returns per-animal predictions and decision scores.
    """
    uniq = np.unique(groups)
    pred = np.zeros(len(y), int)
    score = np.zeros(len(y), float)
    for g in uniq:
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = 0
            continue
        idx = select_features(X[tr], y[tr], k)
        sc = StandardScaler().fit(X[tr][:, idx])
        clf = LDA().fit(sc.transform(X[tr][:, idx]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te][:, idx]))
        score[te] = clf.decision_function(sc.transform(X[te][:, idx]))
    return pred, score


def metrics(y, pred, score):
    acc = float((y == pred).mean())
    sens = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    spec = float(1 - pred[y == 0].mean()) if (y == 0).any() else np.nan
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        auc = np.nan
    return dict(acc=acc, sens=sens, spec=spec, auc=auc, bal=(sens + spec) / 2)


def permutation_test(groups, X, y, k, n_perm=2000, seed=0):
    """Null distribution of balanced accuracy under animal-level label shuffling."""
    rng = np.random.default_rng(seed)
    obs = metrics(y, *lofo_predict(groups, X, y, k))['bal']
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = metrics(yp, *lofo_predict(groups, X, yp, k))['bal']
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, null, p


# ----------------------------------------------------------------- main
if __name__ == '__main__':
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fish, Xa, ya = load('animal')
    out = {}

    if what in ('all', 'pca'):
        Z = StandardScaler().fit_transform(Xa)
        p = PCA().fit(Z)
        S = p.transform(Z)
        out['evr'] = p.explained_variance_ratio_.tolist()
        out['cum'] = np.cumsum(p.explained_variance_ratio_).tolist()
        out['scores'] = S[:, :3].tolist()
        out['fish'] = list(fish)
        out['y'] = ya.tolist()
        # loadings on the first two components
        out['load'] = {FEATURES[i]: [float(p.components_[0, i]),
                                     float(p.components_[1, i])]
                       for i in range(len(FEATURES))}
        km = KMeans(2, n_init=50, random_state=0).fit(S[:, :2])
        lab = km.labels_
        agree = max((lab == ya).mean(), (1 - lab == ya).mean())
        out['kmeans_agreement'] = float(agree)
        print('PC1-2 variance: %.1f%%  |  k-means agreement %d/%d'
              % (100 * sum(p.explained_variance_ratio_[:2]),
                 round(agree * len(ya)), len(ya)))

    if what in ('all', 'clf'):
        res = {}
        for k in (1, 2, 3, 5, 8):
            pred, score = lofo_predict(fish, Xa, ya, k)
            res[k] = metrics(ya, pred, score)
            print(f'k={k}: acc={res[k]["acc"]:.3f} sens={res[k]["sens"]:.2f} '
                  f'spec={res[k]["spec"]:.2f} bal={res[k]["bal"]:.3f} '
                  f'auc={res[k]["auc"]:.3f}')
        out['lofo'] = {str(k): v for k, v in res.items()}
        best = max(res, key=lambda k: res[k]['bal'])
        obs, null, p = permutation_test(fish, Xa, ya, best, n_perm=2000)
        out['perm'] = dict(k=best, observed=obs, p=float(p),
                           null_mean=float(null.mean()),
                           null_q95=float(np.quantile(null, 0.95)))
        print(f'\npermutation test (k={best}): balanced acc={obs:.3f}, '
              f'null mean={null.mean():.3f}, 95th pct={np.quantile(null,0.95):.3f}, '
              f'p={p:.4f}')
        np.save('perm_null.npy', null)

    json.dump(out, open('classify_results.json', 'w'), indent=1)


def permutation_test_full(groups, X, y, ks=(1, 2, 3, 5, 8), n_perm=2000, seed=0):
    """
    Valid null distribution for the *entire* procedure, including the choice of
    how many features to keep. For every permutation the same sweep over k is
    performed and the best balanced accuracy retained, so that the optimism
    introduced by choosing k is present under the null as well.
    """
    rng = np.random.default_rng(seed)

    def best_bal(groups, X, yy):
        return max(metrics(yy, *lofo_predict(groups, X, yy, k))['bal']
                   for k in ks)

    obs = best_bal(groups, X, y)
    null = np.array([best_bal(groups, X, rng.permutation(y))
                     for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, null, p
