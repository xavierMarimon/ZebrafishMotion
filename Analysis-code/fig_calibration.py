"""
Figure 3, rebuilt so that everything except the four frames is vector.

The four video frames come from the two bitmaps embedded in the original
metafile; the arrows, the red outlines, the connecting lines and every label
are redrawn here instead of being reproduced as pixels.

Panel geometry is taken from the original figure, rendered at 200 dpi
(1490 x 351 px for a 536.16 x 126.18 pt figure), so the arrangement matches
the published one:
  frame 1   x  25-417  y  53-303
  frame 2   x 489-886  y  51-303
  frame 3   x 958-1214 y  49-307      outlined in red
  frame 4   x 1293-1483 y  81-281     outlined in red
"""
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

# Fig. 5 is 7.1 in wide and this figure 6.99 in, both included at full width,
# so sizes here are scaled accordingly to match Fig. 5 on the page.
K = 6.99 / 7.1
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8 * K,
                     'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

STRIP = 'src_img/cal-000.png'       # frames 1-3
SMALL = 'src_img/cal-001.png'       # frame 4
CROPS = [(28, 2, 586, 357), (689, 0, 1254, 357), (1359, 0, 1719, 358)]
RED = '#e8112d'

DPI = 200.0                          # the reference render
BOXES = [(25, 53, 417, 303), (489, 51, 886, 303),
         (958, 49, 1214, 307), (1293, 81, 1483, 281)]
TOP_LABELS = ['Initial view', 'Manually focused', 'Cropped', 'Downsampled']
OUTLINED = (2, 3)                    # frames drawn with a red outline


def main():
    strip = np.array(Image.open(STRIP).convert('RGB'))
    imgs = [strip[y0:y1, x0:x1] for x0, y0, x1, y1 in CROPS]
    imgs.append(np.array(Image.open(SMALL).convert('RGB')))

    scale = 7.1 / (1490 / DPI)                    # fit the text width
    box = [[v / DPI * scale for v in b] for b in BOXES]
    xs = [b[0] for b in box] + [b[2] for b in box]
    ys = [b[1] for b in box] + [b[3] for b in box]
    x_off = -min(xs) + 0.02
    LAB_TOP, LAB_BOT = 0.20, 0.24
    FW = max(xs) + x_off + 0.02
    FH = (max(ys) - min(ys)) + LAB_TOP + LAB_BOT + 0.06
    y_off = LAB_BOT + 0.03
    fig = plt.figure(figsize=(FW, FH))

    def place(x0, y0, x1, y1):
        """Rectangle given in reference inches, measured from the top."""
        w, h = x1 - x0, y1 - y0
        yb = FH - y_off - (y1 - min(ys)) - LAB_TOP
        return fig.add_axes([(x0 + x_off) / FW, (yb + y_off) / FH,
                             w / FW, h / FH]), (x0 + x_off, yb + y_off, w, h)

    rects = []
    for k, (b, im) in enumerate(zip(box, imgs)):
        ax, rect = place(*b)
        rects.append(rect)
        ax.imshow(im, interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        if k in OUTLINED:
            for sp in ax.spines.values():
                sp.set_linewidth(2.0); sp.set_color(RED)
        else:
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color('0.45')
        x, y, w, h = rect
        fig.text((x + w / 2) / FW, (y + h + 0.075) / FH, TOP_LABELS[k],
                 ha='center', va='bottom', fontsize=9 * K)

    # ---- block arrows between consecutive frames
    for a, b in zip(rects[:-1], rects[1:]):
        x0 = a[0] + a[2]
        x1 = b[0]
        yc = a[1] + a[3] / 2
        pad = 0.16 * (x1 - x0)
        fig.patches.append(FancyArrow(
            (x0 + pad) / FW, yc / FH, (x1 - x0 - 2 * pad) / FW, 0,
            width=0.055 / FH, head_width=0.155 / FH,
            head_length=0.42 * (x1 - x0 - 2 * pad) / FW,
            length_includes_head=True, color='k', transform=fig.transFigure,
            figure=fig))

    # ---- the two lines that relate the cropped and the down-sampled frame
    c, d = rects[2], rects[3]
    for ya, yb in ((c[1] + c[3], d[1] + d[3]), (c[1], d[1])):
        fig.lines.append(plt.Line2D(
            [(c[0] + c[2]) / FW, d[0] / FW], [ya / FH, yb / FH],
            color='0.25', lw=0.7, transform=fig.transFigure, figure=fig))

    fig.text((rects[0][0] + rects[0][2] / 2) / FW, 0.02, 'Input',
             ha='center', va='bottom', fontsize=8 * K)
    fig.text((rects[3][0] + rects[3][2] / 2) / FW, 0.02, 'Output',
             ha='center', va='bottom', fontsize=8 * K)

    fig.savefig('fig_calibration_orig.pdf')
    plt.close(fig)
    print('ok fig_calibration_orig.pdf  %.2f x %.2f in  frames %s'
          % (FW, FH, [im.shape[:2] for im in imgs]))


if __name__ == '__main__':
    main()
