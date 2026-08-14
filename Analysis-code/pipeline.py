"""
Per-trial feature extraction for the zebrafish locomotion dataset.

Differences from the original MATLAB pipeline, and the reasons:

* No 0.6 Hz low-pass.  The original called MATLAB's ``lowpass(x, 0.6, fs)``,
  i.e. a 0.6 Hz passband edge at fs ~ 115 Hz, discarding >99% of the
  spectrum.  Low-pass filtering is known to corrupt correlation-dimension
  and Lyapunov estimates and to manufacture spurious low-dimensional
  structure (Theiler & Eubank 1993, Chaos 3:771; Badii et al. 1988,
  PRL 60:979).  Here DLC median-filtered coordinates are used directly and
  derivatives are taken with a Savitzky-Golay differentiator.

* Primary analysis signal is the axial bending angle theta(t) between the
  head->tail-base and tail-base->tail-tip segments.  It measures the body
  wave directly - the variable a spinal deformity acts on - and is
  invariant to the tank reference frame.  Head speed v(t) is retained for
  kinematics and as a robustness check.

* Embedding parameters are estimated per trial (AMI for tau, FNN for m) and
  the identical parameters are reused for that trial's surrogates.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch

import nld

PX_PER_CM = 22.3902
DURATION_S = 30.0
LIK_THRESH = 0.90
N_SURR = 39
RQA_MAX_N = 1500
MARKERS = ('Head', 'Tail_B', 'Tail_E')


# ----------------------------------------------------------------------------
def load_points(h5path):
    """Interpolated, calibrated marker coordinates plus quality metadata."""
    df = pd.read_hdf(h5path)
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        df.columns = df.columns.droplevel(0)
    fs = len(df) / DURATION_S
    pts, conf = {}, {}
    for p in MARKERS:
        x = df[(p, 'x')].to_numpy(float)
        y = df[(p, 'y')].to_numpy(float)
        L = df[(p, 'likelihood')].to_numpy(float)
        bad = L < LIK_THRESH
        idx = np.arange(len(x))
        if bad.any() and (~bad).sum() > 10:
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
            y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
        pts[p] = (x / PX_PER_CM, y / PX_PER_CM)
        conf[p] = 1.0 - bad.mean()
    return pts, fs, conf


def _odd_window(fs, seconds, minimum=5):
    w = int(round(seconds * fs))
    w = w + 1 if w % 2 == 0 else w
    return max(w, minimum)


def head_kinematics(pts, fs):
    x, y = pts['Head']
    w = _odd_window(fs, 0.10, 7)
    vx = savgol_filter(x, w, 2, deriv=1, delta=1 / fs)
    vy = savgol_filter(y, w, 2, deriv=1, delta=1 / fs)
    ax = savgol_filter(x, w, 2, deriv=2, delta=1 / fs)
    ay = savgol_filter(y, w, 2, deriv=2, delta=1 / fs)
    v = np.hypot(vx, vy)
    a = np.hypot(ax, ay)
    return dict(distance_cm=float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
                mean_speed=float(v.mean()),
                max_speed=float(np.percentile(v, 99.5)),
                max_accel=float(np.percentile(a, 99.5))), v


def bending_angle(pts, fs):
    """Signed axial bending angle theta(t) in degrees."""
    (hx, hy), (bx, by), (ex, ey) = pts['Head'], pts['Tail_B'], pts['Tail_E']
    v1 = np.c_[bx - hx, by - hy]
    v2 = np.c_[ex - bx, ey - by]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(1)
    theta = np.degrees(np.arctan2(cross, dot))
    w = _odd_window(fs, 0.05, 5)
    return savgol_filter(theta, w, 2)


def spectral(u, fs):
    f, P = welch(u - u.mean(), fs=fs, nperseg=min(1024, len(u)))
    c = np.cumsum(P) / P.sum()
    return dict(f_peak=float(f[np.argmax(P[1:]) + 1]),
                f95=float(f[np.searchsorted(c, 0.95)]))


# ----------------------------------------------------------------------------
def nonlinear_measures(u, fs, tau=None, m=None, full=True):
    """Nonlinear descriptors of a scalar series u(t)."""
    u = np.asarray(u, float)
    if u.std() == 0 or not np.all(np.isfinite(u)):
        return {}
    out = {}
    if tau is None:
        ami = nld.average_mutual_information(u, max_tau=min(200, len(u) // 8))
        tau = max(1, nld.first_min_tau(ami))
    if m is None:
        frac = nld.false_nearest_neighbours(u, tau, max_m=8, nref=500,
                                            theiler=int(fs / 10))
        m = max(2, nld.choose_m(frac))
        out['FNN_min'] = float(np.nanmin(frac))
        out['FNN_m3'] = float(frac[2]) if len(frac) > 2 else np.nan
    out.update(tau=int(tau), m=int(m))

    theiler = max(int(fs / 2), m * tau)
    X = nld.embed(u, m, tau)
    out['theiler'] = int(theiler)

    out['SampEn'] = float(nld.sample_entropy(u[:2500], m=2, r=0.2))
    if full:
        out['ApEn'] = float(nld.approximate_entropy(u[:2500], m=2, r=0.2))

    D2, rs, C, win = nld.correlation_dimension(X, theiler=theiler, nref=450)
    out['D2'] = float(D2) if np.isfinite(D2) else np.nan

    kmax = min(len(X) // 6, int(4.0 * fs))
    t, y = nld.rosenstein(X, fs=fs, theiler=theiler, k_max=kmax, nref=450)
    lam, lwin = nld.lyapunov_auto(t, y)
    out['lambda1'] = float(lam) if np.isfinite(lam) else np.nan

    step = max(1, len(X) // RQA_MAX_N)
    Xr = X[::step][:RQA_MAX_N]
    r = nld.rqa(Xr, rr_target=0.05, theiler=max(1, theiler // step))
    out.update({k: float(r[k]) for k in ('DET', 'LAM', 'TT', 'ENTR', 'Lmax')})

    if full:
        out['_curves'] = dict(rs=rs, C=C, cwin=win, t=t, y=y, lwin=lwin,
                              X=X, theiler=theiler)
    return out


def surrogate_test(u, fs, tau, m, n_surr=N_SURR, seed=0):
    keys = ('SampEn', 'DET', 'ENTR', 'D2')
    dist = {k: [] for k in keys}
    for s in range(n_surr):
        us = nld.iaaft(u, n_iter=120, rng=seed * 1000 + s)
        r = nonlinear_measures(us, fs, tau=tau, m=m, full=False)
        for k in keys:
            dist[k].append(r.get(k, np.nan))
    return {k: np.array(v, float) for k, v in dist.items()}


# alternative hypothesis for each statistic: nonlinear deterministic
# structure lowers entropy / raises determinism relative to linear surrogates
ALT = dict(SampEn='less', DET='greater', ENTR='greater', D2='less')


def process_trial(h5path, fish, trial, do_surrogates=True, seed=0,
                  keep_curves=False):
    pts, fs, conf = load_points(h5path)
    kin, v = head_kinematics(pts, fs)
    theta = bending_angle(pts, fs)

    res = dict(fish=fish, trial=trial, fs=fs, n=len(theta),
               conf_head=conf['Head'], conf_tailB=conf['Tail_B'],
               conf_tailE=conf['Tail_E'], **kin)
    res['theta_sd'] = float(theta.std())
    res['theta_range'] = float(np.percentile(theta, 97.5) -
                               np.percentile(theta, 2.5))
    sp = spectral(theta, fs)
    res['theta_fpeak'] = sp['f_peak']
    res['theta_f95'] = sp['f95']

    nl = nonlinear_measures(theta, fs)
    curves = nl.pop('_curves', None)
    res.update(nl)

    # robustness check on head speed
    nlv = nonlinear_measures(v, fs, full=False)
    for k in ('SampEn', 'DET', 'D2'):
        res['v_' + k] = nlv.get(k, np.nan)

    if do_surrogates and np.isfinite(res.get('SampEn', np.nan)):
        sur = surrogate_test(theta, fs, res['tau'], res['m'], seed=seed)
        for k, d in sur.items():
            res['p_' + k] = nld.surrogate_rank_p(res.get(k, np.nan), d, ALT[k])
            res['sur_' + k + '_mean'] = float(np.nanmean(d))
            res['sur_' + k + '_sd'] = float(np.nanstd(d))
    return (res, curves) if keep_curves else (res, None)


def find_trials(root):
    out = []
    for fish in sorted(os.listdir(root)):
        if fish == 'network_intraerror':
            continue
        fdir = os.path.join(root, fish)
        if not os.path.isdir(fdir):
            continue
        for tr in sorted(os.listdir(fdir), key=lambda s: (len(s), s)):
            tdir = os.path.join(fdir, tr)
            if not os.path.isdir(tdir):
                continue
            h5 = [p for p in glob.glob(os.path.join(tdir, '*_filtered.h5'))
                  if not os.path.basename(p).startswith('._')]
            if h5:
                out.append((fish, tr, h5[0]))
    return out
