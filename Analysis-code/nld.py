"""
Nonlinear time-series analysis toolbox.
Implementations follow the original literature; validated against systems with
known invariants (see validate_nld.py).

References:
  Fraser & Swinney (1986) PRA 33:1134      - mutual information for tau
  Kennel et al. (1992) PRA 45:3403         - false nearest neighbours for m
  Rosenstein et al. (1993) Physica D 65:117- largest Lyapunov exponent
  Grassberger & Procaccia (1983) PRL 50:346- correlation dimension
  Theiler (1986) PRA 34:2427               - Theiler window (temporal correlation)
  Richman & Moorman (2000) AJP 278:H2039   - sample entropy
  Pincus (1991) PNAS 88:2297               - approximate entropy
  Marwan et al. (2007) Phys Rep 438:237    - recurrence quantification analysis
  Schreiber & Schmitz (1996) PRL 77:635    - IAAFT surrogates
"""
import numpy as np


# ----------------------------------------------------------------------------
# Embedding parameters
# ----------------------------------------------------------------------------
def average_mutual_information(x, max_tau=100, nbins=32):
    """AMI(tau) via histogram estimator (Fraser & Swinney 1986)."""
    x = np.asarray(x, float)
    n = len(x)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros(max_tau + 1)
    edges = np.linspace(lo, hi, nbins + 1)
    sym = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)
    ami = np.empty(max_tau + 1)
    for tau in range(max_tau + 1):
        a, b = sym[:n - tau], sym[tau:]
        joint = np.histogram2d(a, b, bins=[nbins, nbins],
                               range=[[0, nbins], [0, nbins]])[0]
        joint /= joint.sum()
        pa = joint.sum(1)
        pb = joint.sum(0)
        nz = joint > 0
        outer = np.outer(pa, pb)
        ami[tau] = np.sum(joint[nz] * np.log(joint[nz] / outer[nz]))
    return ami


def first_min_tau(ami, fallback=None):
    """First local minimum of the AMI curve."""
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] <= ami[i + 1]:
            return i
    return fallback if fallback is not None else int(np.argmin(ami))


def embed(x, m, tau):
    """Time-delay embedding -> (N, m) array."""
    x = np.asarray(x, float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError("series too short for m=%d, tau=%d" % (m, tau))
    return np.column_stack([x[i * tau:i * tau + N] for i in range(m)])


def false_nearest_neighbours(x, tau, max_m=10, rtol=15.0, atol=2.0,
                             theiler=1, nref=1000, rng=None):
    """Fraction of false nearest neighbours vs m (Kennel et al. 1992)."""
    rng = np.random.default_rng(0 if rng is None else rng)
    x = np.asarray(x, float)
    sd = x.std()
    frac = np.empty(max_m)
    for mi, m in enumerate(range(1, max_m + 1)):
        try:
            X = embed(x, m, tau)
        except ValueError:
            frac[mi] = np.nan
            continue
        N = len(X)
        # points that still exist after adding one more coordinate
        usable = N - tau
        if usable < 50:
            frac[mi] = np.nan
            continue
        idx = np.arange(usable)
        if usable > nref:
            idx = rng.choice(usable, nref, replace=False)
        Xs = X[:usable]
        false_cnt = 0
        total = 0
        # chunked nearest-neighbour search with Theiler exclusion
        for start in range(0, len(idx), 200):
            sel = idx[start:start + 200]
            d = np.linalg.norm(Xs[sel][:, None, :] - Xs[None, :, :], axis=2)
            for r, i in enumerate(sel):
                lo, hi = max(0, i - theiler), min(usable, i + theiler + 1)
                d[r, lo:hi] = np.inf
            j = np.argmin(d, axis=1)
            dmin = d[np.arange(len(sel)), j]
            ok = np.isfinite(dmin) & (dmin > 0)
            if not ok.any():
                continue
            i_ok, j_ok, dmin_ok = sel[ok], j[ok], dmin[ok]
            # extra coordinate
            dx = np.abs(x[i_ok + m * tau] - x[j_ok + m * tau])
            crit1 = dx / dmin_ok > rtol
            dnew = np.sqrt(dmin_ok ** 2 + dx ** 2)
            crit2 = dnew / sd > atol
            false_cnt += np.sum(crit1 | crit2)
            total += len(dmin_ok)
        frac[mi] = false_cnt / total if total else np.nan
    return frac


def choose_m(frac, thresh=0.01, max_m=10):
    """Smallest m whose FNN fraction drops below `thresh`."""
    for i, f in enumerate(frac):
        if np.isfinite(f) and f < thresh:
            return i + 1
    return int(np.nanargmin(frac)) + 1


# ----------------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------------
def _pairwise_to_refs(X, ref_idx):
    """Euclidean distances from reference points to all points, float32."""
    A = X[ref_idx].astype(np.float32)
    B = X.astype(np.float32)
    d2 = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :]
          - 2.0 * (A @ B.T))
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2, out=d2)


def correlation_sum(X, theiler=1, nref=500, n_r=40, rng=0):
    """Correlation integral C(r) (Grassberger-Procaccia, Theiler-corrected)."""
    rng = np.random.default_rng(rng)
    N = len(X)
    ref = np.arange(N) if N <= nref else np.sort(rng.choice(N, nref, replace=False))
    d = _pairwise_to_refs(X, ref)
    mask = np.ones_like(d, dtype=bool)
    for r, i in enumerate(ref):
        lo, hi = max(0, i - theiler), min(N, i + theiler + 1)
        mask[r, lo:hi] = False
    dv = d[mask]
    dv = dv[dv > 0]
    if dv.size < 100:
        return None, None
    rmin, rmax = np.percentile(dv, 0.5), dv.max()
    if rmin <= 0:
        rmin = dv[dv > 0].min()
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([(dv < r).mean() for r in rs])
    ok = C > 0
    return rs[ok], C[ok]


def scaling_slope(rs, C, min_pts=6):
    """
    Best-fit power-law exponent of C(r) on the widest window whose local
    slope is most constant (plateau of d log C / d log r).
    Returns (slope, (i0, i1), local_slopes).
    """
    if rs is None or len(rs) < min_pts + 2:
        return np.nan, None, None
    lr, lC = np.log(rs), np.log(C)
    loc = np.gradient(lC, lr)
    best, best_win = np.inf, None
    for i in range(0, len(lr) - min_pts + 1):
        for j in range(i + min_pts, len(lr) + 1):
            s = loc[i:j]
            v = np.std(s) / (np.abs(np.mean(s)) + 1e-12)
            v = v / np.sqrt(j - i)          # favour wider windows
            if v < best:
                best, best_win = v, (i, j)
    i, j = best_win
    slope = np.polyfit(lr[i:j], lC[i:j], 1)[0]
    return slope, best_win, loc


def correlation_dimension(X, theiler=1, nref=500, rng=0):
    rs, C = correlation_sum(X, theiler=theiler, nref=nref, rng=rng)
    D2, win, _ = scaling_slope(rs, C)
    return D2, rs, C, win


def rosenstein(X, fs, theiler=None, k_max=None, nref=500, rng=0):
    """
    Mean logarithmic divergence of nearest neighbours (Rosenstein et al. 1993).
    Returns (times, mean_log_divergence).
    """
    rng = np.random.default_rng(rng)
    N = len(X)
    if theiler is None:
        theiler = max(1, N // 100)
    if k_max is None:
        k_max = min(N // 10, int(2 * fs))
    ref = np.arange(N) if N <= nref else np.sort(rng.choice(N - k_max, min(nref, N - k_max), replace=False))
    ref = ref[ref + k_max < N]
    if len(ref) < 20:
        return None, None
    d = _pairwise_to_refs(X, ref)
    for r, i in enumerate(ref):
        lo, hi = max(0, i - theiler), min(N, i + theiler + 1)
        d[r, lo:hi] = np.inf
    d[:, N - k_max:] = np.inf          # neighbour must be trackable
    j = np.argmin(d, axis=1)
    valid = np.isfinite(d[np.arange(len(ref)), j])
    ref, j = ref[valid], j[valid]
    if len(ref) < 20:
        return None, None
    div = np.empty((len(ref), k_max + 1), dtype=np.float32)
    for k in range(k_max + 1):
        dk = np.linalg.norm(X[ref + k] - X[j + k], axis=1)
        div[:, k] = dk
    with np.errstate(divide='ignore'):
        ld = np.log(div)
    ld[~np.isfinite(ld)] = np.nan
    y = np.nanmean(ld, axis=0)
    t = np.arange(k_max + 1) / fs
    return t, y


def lyapunov_from_curve(t, y, fit_frac=(0.02, 0.15)):
    """Slope of the initial linear region of the divergence curve (fixed window)."""
    if t is None:
        return np.nan, None
    n = len(t)
    i0 = max(1, int(fit_frac[0] * n))
    i1 = max(i0 + 5, int(fit_frac[1] * n))
    i1 = min(i1, n)
    sl = np.polyfit(t[i0:i1], y[i0:i1], 1)[0]
    return sl, (i0, i1)


def lyapunov_auto(t, y, min_pts=8, sat_frac=0.80):
    """
    Largest Lyapunov exponent from the divergence curve, fitting the widest
    window that is still linear and lies before saturation of the curve.

    The curve of Rosenstein et al. (1993) rises linearly at a rate lambda_1
    and then saturates once neighbouring trajectories have separated to the
    size of the attractor. We locate saturation as the first crossing of
    `sat_frac` of the total rise, then select within the pre-saturation
    region the fitting window maximising R^2 weighted by window width.
    Returns (lambda1, (i0, i1)).
    """
    if t is None or y is None or len(t) < min_pts + 2:
        return np.nan, None
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if good.sum() < min_pts + 2:
        return np.nan, None
    y0, ypl = np.nanmin(y), np.nanmean(y[int(0.75 * len(y)):])
    if not np.isfinite(ypl) or ypl <= y0:
        return np.nan, None
    thr = y0 + sat_frac * (ypl - y0)
    above = np.flatnonzero(y >= thr)
    i_sat = int(above[0]) if above.size else len(y) - 1
    i_sat = max(min_pts + 1, min(i_sat, len(y) - 1))
    best, best_win = -np.inf, None
    for i in range(1, max(2, i_sat - min_pts + 1)):
        for j in range(i + min_pts, i_sat + 1):
            tt, yy = t[i:j], y[i:j]
            if not np.all(np.isfinite(yy)):
                continue
            p, res = np.polyfit(tt, yy, 1, full=True)[:2]
            ss = np.sum((yy - yy.mean()) ** 2)
            if ss <= 0 or res.size == 0:
                continue
            r2 = 1 - res[0] / ss
            score = r2 * np.sqrt(j - i)
            if score > best:
                best, best_win = score, (i, j, p[0])
    if best_win is None:
        return np.nan, None
    i, j, slope = best_win
    return slope, (i, j)


def sample_entropy(x, m=2, r=0.2, scale=True):
    """SampEn (Richman & Moorman 2000). r is in units of SD when scale=True."""
    x = np.asarray(x, float)
    N = len(x)
    tol = r * x.std() if scale else r
    if tol <= 0:
        return np.nan

    def count(mm):
        Nm = N - mm + 1
        Z = np.column_stack([x[i:i + Nm] for i in range(mm)])
        c = 0
        for s in range(0, Nm, 256):
            blk = Z[s:s + 256]
            d = np.max(np.abs(blk[:, None, :] - Z[None, :, :]), axis=2)
            for rr in range(len(blk)):
                d[rr, s + rr] = np.inf          # exclude self-match
            c += np.sum(d <= tol)
        return c

    B = count(m)
    A = count(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def approximate_entropy(x, m=2, r=0.2, scale=True):
    """ApEn (Pincus 1991); self-matches included."""
    x = np.asarray(x, float)
    N = len(x)
    tol = r * x.std() if scale else r
    if tol <= 0:
        return np.nan

    def phi(mm):
        Nm = N - mm + 1
        Z = np.column_stack([x[i:i + Nm] for i in range(mm)])
        s = 0.0
        for st in range(0, Nm, 256):
            blk = Z[st:st + 256]
            d = np.max(np.abs(blk[:, None, :] - Z[None, :, :]), axis=2)
            s += np.sum(np.log(np.maximum((d <= tol).sum(1), 1) / Nm))
        return s / Nm

    return phi(m) - phi(m + 1)


def rqa(X, rr_target=0.05, theiler=1, l_min=2, v_min=2, nref=None):
    """
    Recurrence quantification analysis at fixed recurrence rate
    (Marwan et al. 2007). Returns dict with RR, DET, LAM, TT, Lmax, ENTR.
    """
    N = len(X)
    d = _pairwise_to_refs(X, np.arange(N))
    iu = np.triu_indices(N, k=theiler + 1)
    eps = np.quantile(d[iu], rr_target)
    R = d <= eps
    # remove Theiler band
    for k in range(-theiler, theiler + 1):
        np.fill_diagonal(R[max(0, -k):, max(0, k):], False)
    RR = R[iu].mean()

    def line_lengths(mat, diagonal):
        lens = []
        if diagonal:
            for k in range(-(N - 2), N - 1):
                if abs(k) <= theiler:
                    continue
                v = np.diagonal(mat, offset=k)
                lens.extend(_runs(v))
        else:
            for c in range(N):
                lens.extend(_runs(mat[:, c]))
        return np.array(lens, dtype=int) if lens else np.array([], dtype=int)

    dl = line_lengths(R, True)
    vl = line_lengths(R, False)
    npts = R[iu].sum()
    det = dl[dl >= l_min].sum() / (2 * npts) if npts else np.nan
    lam = vl[vl >= v_min].sum() / R.sum() if R.sum() else np.nan
    tt = vl[vl >= v_min].mean() if (vl >= v_min).any() else np.nan
    lmax = dl.max() if dl.size else np.nan
    sel = dl[dl >= l_min]
    if sel.size:
        p = np.bincount(sel)[l_min:]
        p = p[p > 0] / p.sum()
        entr = -np.sum(p * np.log(p))
    else:
        entr = np.nan
    return dict(RR=RR, DET=det, LAM=lam, TT=tt, Lmax=lmax, ENTR=entr, EPS=eps)


def _runs(v):
    """Lengths of consecutive True runs in a boolean vector."""
    v = np.asarray(v, bool)
    if not v.any():
        return []
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    return list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))


# ----------------------------------------------------------------------------
# Surrogates
# ----------------------------------------------------------------------------
def iaaft(x, n_iter=200, rng=0):
    """
    Iterative amplitude-adjusted Fourier-transform surrogate
    (Schreiber & Schmitz 1996): preserves the amplitude distribution and,
    approximately, the power spectrum of x, while destroying nonlinear
    (phase) structure.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    s = rng.permutation(x)
    for _ in range(n_iter):
        S = np.fft.rfft(s)
        ph = np.angle(S)
        s = np.fft.irfft(amp * np.exp(1j * ph), n=n)
        ranks = np.argsort(np.argsort(s))
        s_new = sorted_x[ranks]
        if np.allclose(s_new, s, rtol=1e-10, atol=1e-12):
            s = s_new
            break
        s = s_new
    return s


def surrogate_rank_p(observed, surrogates, alternative='two-sided'):
    """
    Rank-based p-value of the observed statistic within the surrogate
    ensemble (Theiler et al. 1992). With M surrogates the smallest
    attainable one-sided p is 1/(M+1).
    """
    sur = np.asarray(surrogates, float)
    sur = sur[np.isfinite(sur)]
    M = len(sur)
    if M == 0 or not np.isfinite(observed):
        return np.nan
    n_ge = np.sum(sur >= observed)
    n_le = np.sum(sur <= observed)
    if alternative == 'greater':
        return (1 + n_ge) / (M + 1)
    if alternative == 'less':
        return (1 + n_le) / (M + 1)
    return min(1.0, 2 * min((1 + n_ge) / (M + 1), (1 + n_le) / (M + 1)))
