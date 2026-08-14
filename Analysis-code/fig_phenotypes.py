"""
Figure 1, rebuilt so that everything except the photographs is vector.

The source is the composite bitmap of the previous manuscript, in which the
red arrows, the scale bar and the "1cm" label were burnt into the raster and
therefore reproduced as pixels. Here the two photographic blocks are cropped
out unchanged, the burnt-in arrows are removed by local median inpainting,
and the arrows, the scale bar, the labels and the panel titles are redrawn as
vector graphics in the typeface used by the rest of the figures.

Geometry measured on the source bitmap (1446 x 408 px):
  left  photographic block  x 142-688, y  37-365   (scoliotic)
  right photographic block  x 800-1345, y 38-362   (healthy)
  burnt-in scale bar        70 px long  =  1 cm
  four red arrows           bounding boxes listed in ARROWS below

The photographs themselves are reproduced at their native resolution; the
source contains no higher-resolution version, so their sharpness is unchanged.
"""
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.titlesize': 8.5, 'figure.dpi': 300,
                     'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

SRC = 'fig_phenotypes_source.png'
LEFT = (142, 37, 688, 365)          # x0, y0, x1, y1  (scoliotic)
RIGHT = (800, 38, 1345, 362)        # healthy
BAR_PX = 70.0                       # length of the burnt-in bar, = 1 cm
ARROW_RED = '#e8112d'

# tail and head of each arrow, in source-bitmap coordinates
ARROWS = [((326, 38), (326, 62)),       # upper photo, pointing down
          ((326, 136), (326, 103)),     # upper photo, pointing up
          ((457, 250), (424, 280)),     # lower photo, pointing down-left
          ((369, 335), (404, 310))]     # lower photo, pointing up-right


def inpaint_red(rgb, n_iter=14):
    """Replace the burnt-in red arrows by a local median of their surroundings."""
    a = rgb.astype(float).copy()
    mask = ((a[:, :, 0] > 140) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100))
    mask = np.pad(mask, 2, mode='constant')
    for _ in range(2):                       # widen slightly to catch the halo
        mask[1:-1, 1:-1] |= (mask[:-2, 1:-1] | mask[2:, 1:-1] |
                             mask[1:-1, :-2] | mask[1:-1, 2:])
    mask = mask[2:-2, 2:-2]
    out = a.copy()
    out[mask] = np.nan
    for _ in range(n_iter):
        todo = np.isnan(out[:, :, 0])
        if not todo.any():
            break
        pad = np.pad(out, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1],
                       pad[1:-1, :-2], pad[1:-1, 2:],
                       pad[:-2, :-2], pad[:-2, 2:],
                       pad[2:, :-2], pad[2:, 2:]])
        fill = np.nanmedian(nb, axis=0)
        out[todo] = fill[todo]
    return np.clip(np.nan_to_num(out, nan=255.0), 0, 255).astype(np.uint8)


def main():
    src = np.array(Image.open(SRC).convert('RGB'))
    left = inpaint_red(src[LEFT[1]:LEFT[3], LEFT[0]:LEFT[2]])
    right = src[RIGHT[1]:RIGHT[3], RIGHT[0]:RIGHT[2]]

    FW = 7.1
    M, GAP = 0.05, 0.34
    PW = (FW - 2 * M - GAP) / 2
    hL = PW * left.shape[0] / left.shape[1]
    hR = PW * right.shape[0] / right.shape[1]
    TITLE, BAR = 0.24, 0.34
    FH = TITLE + max(hL, hR) + BAR + 0.04
    fig = plt.figure(figsize=(FW, FH))

    def place(x_in, y_in, w_in, h_in):
        return fig.add_axes([x_in / FW, y_in / FH, w_in / FW, h_in / FH])

    top = FH - TITLE
    axL = place(M, top - hL, PW, hL)
    axR = place(M + PW + GAP, top - hR, PW, hR)
    for ax, im in ((axL, left), (axR, right)):
        ax.imshow(im, interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color('0.45')

    # ---- arrows, redrawn in vector on the cleaned crop
    for (tx, ty), (hx, hy) in ARROWS:
        axL.annotate('', xy=(hx - LEFT[0], hy - LEFT[1]),
                     xytext=(tx - LEFT[0], ty - LEFT[1]),
                     arrowprops=dict(arrowstyle='-|>,head_width=0.30,'
                                     'head_length=0.42',
                                     color=ARROW_RED, lw=2.6,
                                     shrinkA=0, shrinkB=0,
                                     joinstyle='miter'))

    axL.set_title('(a) scoliotic', loc='left', pad=3)
    axR.set_title('(b) healthy', loc='left', pad=3)

    # ---- scale bar, drawn to the length the burnt-in bar encoded
    frac = BAR_PX / right.shape[1]
    bw = PW * frac
    bx = M + PW + GAP + PW - bw - 0.04
    by = top - hR - 0.15
    axB = place(bx, by, bw, 0.045)
    axB.set_axis_off()
    axB.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axB.transAxes,
                                facecolor='k', edgecolor='none'))
    fig.text((bx + bw / 2) / FW, (by - 0.03) / FH, '1 cm', ha='center',
             va='top', fontsize=8)

    fig.savefig('fig_phenotypes.pdf')
    plt.close(fig)
    print('ok fig_phenotypes.pdf  |  crops %s %s  |  1 cm = %.3f in'
          % (left.shape[:2], right.shape[:2], bw))


if __name__ == '__main__':
    main()
