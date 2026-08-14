"""
Second-generation descriptors: measures that remain valid for short, noisy,
non-stationary, nonlinear-*stochastic* signals, i.e. exactly the regime the
zebrafish bending angle occupies.

References
  Bandt & Pompe (2002) PRL 88:174102        - permutation entropy
  Rosso et al. (2007) PRL 99:154102         - complexity-entropy causality plane
  Lamberti et al. (2004) Physica A 334:119  - statistical complexity normalisation
  Amigo et al. (2007) EPL 79:50001          - forbidden ordinal patterns
  Costa et al. (2002) PRL 89:068102         - multiscale entropy
  Wu et al. (2014) Phys Lett A 378:1369     - refined composite MSE
  Peng et al. (1994) PRE 49:1685            - detrended fluctuation analysis
  Kantelhardt et al. (2002) Physica A 316:87- multifractal DFA
  Donner et al. (2010) New J Phys 12:033025 - recurrence networks
  Schreiber & Schmitz (1997) PRL 79:1475    - time-reversal asymmetry
  Lizier et al. (2012) Front Robot AI       - active information storage
  Schreiber (2000) PRL 85:461               - transfer entropy
"""
import numpy as np
from math import factorial, lgamma


# ----------------------------------------------------------------------------
# Ordinal (permutation) methods
# ----------------------------------------------------------------------------
def _ordinal_symbols(x, d=5, tau=1):
    """Lehmer-coded ordinal patterns of embedded windows."""
    x = np.asarray(x, float)
    N = len(x) - (d - 1) * tau
    if N <= 0:
        return np.array([], dtype=int)
    X = np.column_stack([x[i * tau:i * tau + N] for i in range(d)])
    order = np.argsort(X, axis=1, kind='stable')
    # encode permutation -> integer via factorial number system
    code = np.zeros(N, dtype=np.int64)
    for i in range(d):
        smaller = np.sum(order[:, i + 1:] < order[:, i][:, None], axis=1)
        code += smaller * factorial(d - 1 - i)
    return code


def ordinal_distribution(x, d=5, tau=1):
    codes = _ordinal_symbols(x, d, tau)
    n = factorial(d)
    if codes.size == 0:
        return np.zeros(n)
    p = np.bincount(codes, minlength=n).astype(float)
    return p / p.sum()


def _shannon(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def permutation_entropy(x, d=5, tau=1, normalise=True):
    """Normalised permutation entropy H in [0, 1] (Bandt & Pompe 2002)."""
    p = ordinal_distribution(x, d, tau)
    H = _shannon(p)
    return H / np.log(factorial(d)) if normalise else H


def statistical_complexity(x, d=5, tau=1):
    """
    Jensen-Shannon statistical complexity C_JS and normalised entropy H
    (Rosso et al. 2007; normalisation after Lamberti et al. 2004).
    Returns (H, C).
    """
    P = ordinal_distribution(x, d, tau)
    n = factorial(d)
    Pe = np.full(n, 1.0 / n)
    H = _shannon(P) / np.log(n)
    M = 0.5 * (P + Pe)
    JS = _shannon(M) - 0.5 * _shannon(P) - 0.5 * _shannon(Pe)
    Q0 = -2.0 / (((n + 1.0) / n) * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))
    C = Q0 * JS * H
    return H, C


def complexity_bounds(n, npts=200):
    """
    Lower/upper bounds of C_JS as a function of H for an ordinal alphabet of
    size n (Martin, Plastino & Rosso 2006).  Used to place a system in the
    complexity-entropy plane relative to the chaotic/stochastic regions.
    """
    Q0 = -2.0 / (((n + 1.0) / n) * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))

    def CH(P):
        P = P[P > 0]
        H = _shannon(P) / np.log(n)
        Pe = np.full(n, 1.0 / n)
        M = 0.5 * (np.pad(P, (0, n - len(P))) + Pe)
        JS = _shannon(M) - 0.5 * _shannon(P) - 0.5 * _shannon(Pe)
        return H, Q0 * JS * H

    # upper bound: one dominant state with probability p, rest uniform
    up = []
    for p in np.linspace(1.0 / n + 1e-9, 1 - 1e-9, npts):
        P = np.concatenate(([p], np.full(n - 1, (1 - p) / (n - 1))))
        up.append(CH(P))
    # lower bound: k states uniformly populated
    lo = []
    for k in range(1, n + 1):
        P = np.full(k, 1.0 / k)
        lo.append(CH(P))
    return np.array(sorted(lo)), np.array(sorted(up))


def missing_patterns(x, d=5, tau=1):
    """Fraction of ordinal patterns never observed (Amigo et al. 2007)."""
    p = ordinal_distribution(x, d, tau)
    n = len(x) - (d - 1) * tau
    if n < factorial(d) * 5:      # not enough data to call a pattern forbidden
        return np.nan
    return float(np.mean(p == 0))


# ----------------------------------------------------------------------------
# Multiscale entropy
# ----------------------------------------------------------------------------
def _sampen_counts(x, m, r):
    """Template match counts (B, A) for sample entropy."""
    N = len(x)
    if N < m + 2:
        return 0, 0
    def cnt(mm):
        Nm = N - mm + 1
        Z = np.column_stack([x[i:i + Nm] for i in range(mm)])
        c = 0
        for s in range(0, Nm, 256):
            blk = Z[s:s + 256]
            dd = np.max(np.abs(blk[:, None, :] - Z[None, :, :]), axis=2)
            for rr in range(len(blk)):
                dd[rr, s + rr] = np.inf
            c += int(np.sum(dd <= r))
        return c
    return cnt(m), cnt(m + 1)


def rcmse(x, scales=None, m=2, r=0.15, max_len=2500):
    """
    Refined composite multiscale entropy (Wu et al. 2014): at each scale the
    match counts are pooled over all coarse-graining offsets before taking the
    logarithm, which greatly reduces the variance of classical MSE on short
    records.  r is fixed to the SD of the *original* series (Costa et al.).
    """
    x = np.asarray(x, float)[:max_len]
    if scales is None:
        scales = list(range(1, 16))
    tol = r * x.std()
    out = {}
    for s in scales:
        if len(x) // s < 50:
            out[s] = np.nan
            continue
        B = A = 0
        for k in range(s):
            y = x[k:]
            L = (len(y) // s) * s
            cg = y[:L].reshape(-1, s).mean(axis=1)
            b, a = _sampen_counts(cg, m, tol)
            B += b
            A += a
        out[s] = -np.log(A / B) if (A > 0 and B > 0) else np.nan
    return out


def mse_summary(curve):
    """Complexity index (area under the MSE curve) and slope over scales."""
    s = np.array([k for k, v in curve.items() if np.isfinite(v)], float)
    v = np.array([v for v in curve.values() if np.isfinite(v)], float)
    if len(v) < 4:
        return dict(CI=np.nan, MSE_slope=np.nan, MSE_s1=np.nan, MSE_peak=np.nan)
    return dict(CI=float(np.trapezoid(v, s) / (s[-1] - s[0])),
                MSE_slope=float(np.polyfit(s, v, 1)[0]),
                MSE_s1=float(v[0]),
                MSE_peak=float(s[np.argmax(v)]))


# ----------------------------------------------------------------------------
# Scaling / multifractality
# ----------------------------------------------------------------------------
def _dfa_fluctuations(x, scales, q_list, order=1):
    """Generalised DFA fluctuation functions F_q(s) (Kantelhardt et al. 2002)."""
    x = np.asarray(x, float)
    Y = np.cumsum(x - x.mean())
    N = len(Y)
    F = {q: [] for q in q_list}
    used = []
    for s in scales:
        ns = N // s
        if ns < 4:
            continue
        var = []
        for seq in (Y[:ns * s], Y[N - ns * s:]):        # forward and backward
            seg = seq.reshape(ns, s)
            t = np.arange(s)
            coef = np.polyfit(t, seg.T, order)
            trend = np.polyval(coef, t[:, None]).T
            var.append(np.mean((seg - trend) ** 2, axis=1))
        var = np.concatenate(var)
        var = var[var > 0]
        if var.size < 4:
            continue
        used.append(s)
        for q in q_list:
            if abs(q) < 1e-8:
                F[q].append(np.exp(0.25 * np.mean(np.log(var))))
            else:
                F[q].append(np.mean(var ** (q / 2.0)) ** (1.0 / q))
    return np.array(used, float), {q: np.array(v, float) for q, v in F.items()}


def dfa(x, scales=None, order=1):
    """Monofractal DFA exponent alpha."""
    N = len(x)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(N // 8), 18).astype(int))
    s, F = _dfa_fluctuations(x, scales, [2.0], order)
    if len(s) < 5:
        return np.nan
    return float(np.polyfit(np.log(s), np.log(F[2.0]), 1)[0])


def mfdfa(x, scales=None, q_list=None, order=1):
    """
    Multifractal DFA.  Returns the generalised Hurst exponents h(q), the
    singularity spectrum width Delta-alpha, and its asymmetry.
    """
    N = len(x)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(16), np.log10(N // 8), 16).astype(int))
    if q_list is None:
        q_list = [-5, -3, -2, -1, 0.0001, 1, 2, 3, 5]
    s, F = _dfa_fluctuations(x, scales, q_list, order)
    if len(s) < 5:
        return dict(alpha_width=np.nan, alpha_asym=np.nan, h2=np.nan, dh=np.nan)
    hq = np.array([np.polyfit(np.log(s), np.log(F[q]), 1)[0] for q in q_list])
    q = np.array(q_list, float)
    tau = q * hq - 1
    alpha = np.gradient(tau, q)
    f = q * alpha - tau
    return dict(alpha_width=float(alpha.max() - alpha.min()),
                alpha_asym=float((alpha[np.argmax(f)] - alpha.min()) /
                                 (alpha.max() - alpha.min() + 1e-12)),
                h2=float(hq[list(q_list).index(2)]),
                dh=float(hq.max() - hq.min()))


# ----------------------------------------------------------------------------
# Recurrence networks
# ----------------------------------------------------------------------------
def recurrence_network(X, rr_target=0.05, theiler=1, max_n=1200):
    """
    Complex-network measures of the recurrence matrix (Donner et al. 2010).
    These characterise phase-space geometry without requiring attractor
    invariants to exist.
    """
    step = max(1, len(X) // max_n)
    Xs = X[::step][:max_n]
    n = len(Xs)
    A = Xs.astype(np.float32)
    d2 = (np.sum(A * A, 1)[:, None] + np.sum(A * A, 1)[None, :] - 2 * (A @ A.T))
    np.maximum(d2, 0, out=d2)
    d = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    eps = np.quantile(d[iu], rr_target)
    R = (d <= eps)
    np.fill_diagonal(R, False)
    th = max(1, theiler // step)
    for k in range(-th, th + 1):
        np.fill_diagonal(R[max(0, -k):, max(0, k):], False)
    A = R.astype(np.float32)
    deg = A.sum(1)
    # transitivity = 3 x triangles / connected triples
    tri = np.trace(A @ A @ A)
    triples = np.sum(deg * (deg - 1))
    trans = float(tri / triples) if triples > 0 else np.nan
    # local clustering -> average clustering coefficient
    with np.errstate(invalid='ignore', divide='ignore'):
        A2 = A @ A
        num = np.einsum('ij,ij->i', A2, A)
        den = deg * (deg - 1)
        cl = np.where(den > 0, num / den, np.nan)
    return dict(RN_trans=trans,
                RN_clust=float(np.nanmean(cl)),
                RN_degcv=float(np.std(deg) / (np.mean(deg) + 1e-12)),
                RN_eps=float(eps))


# ----------------------------------------------------------------------------
# Nonlinearity / asymmetry statistics
# ----------------------------------------------------------------------------
def time_reversal_asymmetry(x, lag=1):
    """
    T_rev = <(x_{t+l} - x_t)^3> / <(x_{t+l} - x_t)^2>^{3/2}
    (Schreiber & Schmitz 1997).  Zero for any linear Gaussian process and for
    any time-reversible process; a powerful discriminating statistic against
    IAAFT surrogates.
    """
    x = np.asarray(x, float)
    dx = x[lag:] - x[:-lag]
    m2 = np.mean(dx ** 2)
    if m2 <= 0:
        return np.nan
    return float(np.mean(dx ** 3) / m2 ** 1.5)


def lateral_asymmetry(theta, fs, min_amp=5.0):
    """
    Left-right symmetry of the axial bending angle.

    A lateral spinal curvature should bias the body wave towards the convex
    side.  We therefore quantify (i) the static offset of theta, (ii) the
    skewness of its distribution, and (iii) half-cycle asymmetry: excursions
    either side of the median are compared in amplitude, duration and count.
    """
    th = np.asarray(theta, float)
    med = np.median(th)
    c = th - med
    sd = c.std()
    out = dict(theta_mean=float(th.mean()),
               theta_median=float(med),
               theta_skew=float(np.mean(c ** 3) / (sd ** 3 + 1e-12)),
               theta_kurt=float(np.mean(c ** 4) / (sd ** 4 + 1e-12) - 3.0))

    sign = np.sign(c)
    sign[sign == 0] = 1
    idx = np.flatnonzero(np.diff(sign) != 0) + 1
    segs = np.split(np.arange(len(c)), idx)
    amp_p, amp_n, dur_p, dur_n = [], [], [], []
    for s in segs:
        if s.size < 3:
            continue
        seg = c[s]
        a = np.max(np.abs(seg))
        if a < min_amp:
            continue
        (amp_p if seg.mean() > 0 else amp_n).append(a)
        (dur_p if seg.mean() > 0 else dur_n).append(s.size / fs)

    def _ai(a, b):
        a, b = float(np.mean(a)) if a else np.nan, float(np.mean(b)) if b else np.nan
        if not np.isfinite(a) or not np.isfinite(b) or (a + b) == 0:
            return np.nan
        return (a - b) / (a + b)

    out.update(asym_amp=_ai(amp_p, amp_n),
               asym_dur=_ai(dur_p, dur_n),
               asym_count=_ai([len(amp_p)] if amp_p else [], [len(amp_n)] if amp_n else []),
               n_halfcycles=len(amp_p) + len(amp_n),
               beat_amp=float(np.nanmean(amp_p + amp_n)) if (amp_p or amp_n) else np.nan,
               beat_rate=float((len(amp_p) + len(amp_n)) / (len(th) / fs)))
    return out


# ----------------------------------------------------------------------------
# Information-theoretic measures
# ----------------------------------------------------------------------------
def ami_profile(x, max_lag, nbins=32):
    """Auto mutual information curve and its characteristic decay time."""
    import nld
    a = nld.average_mutual_information(x, max_tau=max_lag, nbins=nbins)
    a0 = a[0]
    below = np.flatnonzero(a <= a0 / np.e)
    return dict(AMI_0=float(a0),
                AMI_decay=float(below[0]) if below.size else np.nan,
                AMI_firstmin=float(nld.first_min_tau(a)))


def _sym(x, d, tau):
    return _ordinal_symbols(x, d, tau)


def active_information_storage(x, d=4, tau=1):
    """
    AIS = I(past ordinal pattern ; next value's ordinal rank), estimated on the
    ordinal alphabet so that it is robust to noise and monotone transforms
    (Lizier et al. 2012).
    """
    codes = _sym(x, d, tau)
    if codes.size < 200:
        return np.nan
    nxt = _sym(x, 2, tau)[(d - 1) * tau:][:len(codes) - 1]
    past = codes[:len(nxt)]
    if past.size < 100:
        return np.nan
    return _mi_discrete(past, nxt)


def _mi_discrete(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    J = np.zeros((len(ua), len(ub)))
    np.add.at(J, (ia, ib), 1.0)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    return float(np.sum(J[nz] * np.log(J[nz] / np.outer(pa, pb)[nz])))


def transfer_entropy(src, dst, d=3, tau=1):
    """
    Ordinal transfer entropy TE(src -> dst) = I(dst_next ; src_past | dst_past),
    estimated with ordinal patterns (Schreiber 2000; Staniek & Lehnertz 2008).
    """
    s = _sym(src, d, tau)
    t = _sym(dst, d, tau)
    n = min(len(s), len(t)) - 1
    if n < 300:
        return np.nan
    tp, sp, tn = t[:n], s[:n], t[1:n + 1]
    # H(tn|tp) - H(tn|tp,sp)
    return _cond_mi(tn, sp, tp)


def _cond_mi(a, b, c):
    """I(a ; b | c) for discrete variables."""
    abc = np.stack([a, b, c], 1)
    def H(cols):
        _, idx = np.unique(abc[:, cols], axis=0, return_inverse=True)
        p = np.bincount(idx).astype(float)
        p /= p.sum()
        return _shannon(p)
    return float(H([0, 2]) + H([1, 2]) - H([0, 1, 2]) - H([2]))
