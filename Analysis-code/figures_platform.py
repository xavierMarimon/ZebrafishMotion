"""Platform / tracking figures, redrawn as vector graphics from source data."""
import glob
import os
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import pipeline as P

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.7, 'savefig.bbox': 'tight',
                     'axes.titlesize': 9, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'figure.dpi': 300})

Z = ("/sessions/tender-affectionate-carson/mnt/Article Zebrafish/"
     "zebrafish material and code")
TRAJ = f"{Z}/TRAJECTORY RESULTS"
CH, CS = '#2c6fb5', '#d1495b'
MARKERS = ('Head', 'Tail_B', 'Tail_E')
CMAP = plt.get_cmap('rainbow')
MCOL = {'Head': CMAP(0.0), 'Tail_B': CMAP(0.5), 'Tail_E': CMAP(1.0)}
FRAME_PX = 480.0


# ---------------------------------------------------------------- fig 5
def marker_precision():
    """
    Per-landmark positional jitter across all trials.

    Repeated DeepLabCut inference on the same video is deterministic to
    ~1e-5 px, so repeatability across runs carries no information about
    tracking precision. The informative quantity is the high-frequency
    positional residual after local smoothing, which we report together with
    the fraction of frames above the confidence threshold.
    """
    return pd.read_csv('marker_jitter.csv')


def fig_network(fname='fig_network_eval.pdf'):
    ls = pd.read_csv(f"{Z}/DLC trained networks/ZebrafishNet/dlc-models/"
                     "iteration-0/ZebrafishNet2.0Apr28-trainset95shuffle1/"
                     "train/learning_stats.csv",
                     header=None, names=['iter', 'loss', 'lr'])
    j = marker_precision()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5),
                             gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    ax.plot(ls['iter'], ls['loss'], color='#c0392b', lw=0.7)
    ax.axvline(25000, color='0.35', ls='--', lw=0.8)
    ax.annotate('snapshot\nused', (25000, ls.loss.max() * 0.72),
                xytext=(-30, 0), textcoords='offset points', fontsize=6.2,
                ha='right', va='center', color='0.25')
    ax.set_xlabel('training iteration')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title('(a) training loss', loc='left')
    ax.set_xlim(0, ls['iter'].max())
    axin = ax.inset_axes([0.40, 0.40, 0.57, 0.45])
    sel = ls['iter'] > 5000
    axin.plot(ls['iter'][sel], ls['loss'][sel], color='#c0392b', lw=0.6)
    axin.axvline(25000, color='0.35', ls='--', lw=0.7)
    axin.set_ylim(0, ls.loss[sel].max() * 1.15)
    axin.tick_params(labelsize=5.5)
    axin.set_title('plateau', fontsize=6)

    ax = axes[1]
    for i, mk in enumerate(MARKERS):
        v = j[mk].values
        ax.scatter(np.full(len(v), i) +
                   np.random.default_rng(i).normal(0, 0.07, len(v)),
                   v, s=6, color=MCOL[mk], alpha=0.55, lw=0, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color='k', lw=1.4,
                zorder=5)
    ax.set_yscale('log')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Head', 'Tail_B', 'Tail_E'])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('positional jitter, RMS (px)')
    ax.set_title('(b) landmark precision (110 trials)', loc='left')
    for i, mk in enumerate(MARKERS):
        ax.text(i, ax.get_ylim()[1] * 0.55, f"{j[mk + '_lik'].mean():.1f}%",
                ha='center', fontsize=6, color='0.3')

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print('ok', fname, {mk: round(float(j[mk].mean()), 3) for mk in MARKERS})


# ---------------------------------------------------------------- fig 6
def fig_pose_output(fname='fig_pose_output.pdf'):
    trials = {(f, t): h for f, t, h in P.find_trials(TRAJ)}
    df = pd.read_hdf(trials[('f2', 'f2v1')])
    if df.columns.nlevels == 3:
        df.columns = df.columns.droplevel(0)
    n = len(df)
    frames = np.arange(n)

    fig = plt.figure(figsize=(7.1, 5.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04],
                          height_ratios=[1.75, 0.85, 0.6],
                          hspace=0.55, wspace=0.28)

    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(mpimg.imread('frame_f2v1_1500.png'))
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title('(a) labelled frame', loc='left')

    axB = fig.add_subplot(gs[0, 1])
    for mk in MARKERS:
        axB.scatter(df[(mk, 'x')], df[(mk, 'y')], s=1.4, color=MCOL[mk],
                    edgecolors='none', alpha=0.8)
    axB.invert_yaxis()
    axB.set_xlabel('X position (px)'); axB.set_ylabel('Y position (px)')
    axB.set_title('(b) reconstructed trajectory', loc='left')

    axC = fig.add_subplot(gs[1, 0:2])
    for mk in MARKERS:
        axC.plot(frames, df[(mk, 'y')], '-', color=MCOL[mk], lw=1.0)
        axC.plot(frames, df[(mk, 'x')], '--', color=MCOL[mk], lw=1.0)
    axC.set_xlim(0, n)
    axC.set_xticklabels([])
    axC.set_ylabel('X (dashed) and\nY (solid) (px)')
    axC.set_title('(c) coordinates over time', loc='left')

    axD = fig.add_subplot(gs[2, 0:2])
    for mk in MARKERS:
        axD.plot(frames, df[(mk, 'likelihood')], '-', color=MCOL[mk], lw=1.0)
    axD.axhline(0.9, color='0.35', ls='--', lw=0.7)
    axD.set_xlim(0, n); axD.set_ylim(0, 1.05)
    axD.set_xlabel('frame index'); axD.set_ylabel('likelihood')
    axD.set_title('(d) per-frame detection confidence', loc='left')

    cax = fig.add_subplot(gs[:, 2])
    sm = cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Head', 'Tail_B', 'Tail_E'])
    fig.savefig(fname)
    plt.close(fig)
    print('ok', fname, 'frames', n)


# ---------------------------------------------------------------- fig 7
def fig_trajectories(fname='fig_trajectories.pdf'):
    """
    Reproduces the original MATLAB trajectory plot (DLC_Plotting_cm.m).

    That script plots Head_cm and Tail_B_cm, i.e. the calibrated but
    *unfiltered* coordinates; the low-pass filtered series were used only for
    feature computation, never for display.
    """
    trials = {(f, t): h for f, t, h in P.find_trials(TRAJ)}
    sel = [(('f8', 'f8v10'), 'healthy (f8, trial 10)', '(a)'),
           (('sf3', 'sf3v10'), 'scoliotic (sf3, trial 10)', '(b)')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, (key, title, lab) in zip(axes, sel):
        pts, fs, _ = P.load_points(trials[key])
        hx, hy = pts['Head']
        bx, by = pts['Tail_B']
        ax.plot(hx, hy, color='#1f4fd8', lw=1.4, alpha=0.9, zorder=2,
                solid_capstyle='round')
        ax.plot(bx, by, color='#d62728', lw=1.4, alpha=0.7, zorder=1,
                solid_capstyle='round')
        for (xx, yy, col) in ((hx, hy, '#1f4fd8'), (bx, by, '#d62728')):
            ax.scatter(xx[0], yy[0], s=58, marker='>', c=col, ec='k', lw=0.7,
                       zorder=6)
            ax.scatter(xx[-1], yy[-1], s=58, marker='o', c=col, ec='k',
                       lw=0.7, zorder=6)
        # room above the arena for the legend, so it never overlaps the path
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 25.5)
        ax.set_yticks(np.arange(0, 21, 5))
        ax.set_aspect('equal')
        ax.grid(True, color='0.88', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel('tank position X (cm)')
        ax.set_title(f'{lab} {title}', loc='left')
    axes[0].set_ylabel('tank position Y (cm)')
    for ax in axes:
        ax.legend(handles=[
            Line2D([], [], color='#1f4fd8', lw=1.6, label='head'),
            Line2D([], [], color='#d62728', lw=1.6, label='tail base'),
            Line2D([], [], marker='>', ls='', mfc='w', mec='k', ms=6,
                   label='start'),
            Line2D([], [], marker='o', ls='', mfc='w', mec='k', ms=6,
                   label='end')],
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4,
            frameon=False, fontsize=6.6, handlelength=1.4,
            columnspacing=1.1, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print('ok', fname)


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'net'):
        fig_network()
    if which in ('all', 'pose'):
        fig_pose_output()
    if which in ('all', 'traj'):
        fig_trajectories()
