"""
Figure 2, rebuilt so that everything except the two photographs is vector.

In the previous manuscript this figure was a screen capture of the DeepLabCut
labelling window: the axes, the tick numbers, the colour key, the three
markers and the callout were all pixels. Here only the microscope image and
the schematic drawing are kept as raster; the axes, ticks, labels, colour
key, markers and callout are redrawn.

Everything was measured on the original screen capture (1226 x 864 px):
  plot frame            rows 44-812, columns 103-1021
  x tick marks          100 px per 10 units, 200 at column 164
  y tick marks          99.6 px per 10 units, 120 at row 127.5
  marker centroids      head (454, 242), tail base (597, 568),
                        tail tip (651, 297)
  colour key            columns 1029-1075
which gives the axis limits and marker coordinates used below. The colour map
is resampled from the original key so that the correspondence between colour
and body landmark is preserved exactly.
"""
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
from scipy import ndimage

# Type size: matched to Fig. 5, which is 7.1 in wide and included at 0.92 of
# the text width. K compensates for this figure being included at 0.80.
K = 0.92 / 0.80
# Picture size: the panels are drawn R times smaller than the type would
# imply, so that the image shrinks while the labels keep their size on the
# page. The LaTeX width is chosen to preserve the scale factor exactly.
R = 0.80 / 0.92
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8 * K,
                     'axes.linewidth': 0.7 * K, 'axes.titlesize': 9 * K,
                     'axes.labelsize': 8 * K, 'xtick.labelsize': 8 * K,
                     'ytick.labelsize': 8 * K, 'figure.dpi': 300,
                     'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

SHOT = 'src_img/dlc-000.png'        # the screen capture
DRAW = 'src_img/dlc-002.png'        # the schematic drawing
DRAW_A = 'src_img/dlc-003.png'      # its transparency mask

FRAME = dict(top=44, bottom=812, left=103, right=1021)
CBAR = (1029, 1075)
X0_PX, X0_VAL, X_SCALE = 164.0, 200.0, 10.0        # px per unit
Y0_PX, Y0_VAL, Y_SCALE = 127.5, 120.0, 9.958
MARKERS_PX = {'Head': (454, 242), 'Tail_B': (597, 568),
              'Tail_E': (651, 297)}
# positions of the three dots on the drawing, as fractions of its bounding box
DRAW_DOTS = {'Head': (0.502, 0.0895), 'Tail_B': (0.431, 0.566),
             'Tail_E': (0.521, 0.941)}


# region of the burnt-in callout, and the burnt-in markers, to be painted out
CALLOUT = (665, 205, 915, 302)      # x0, y0, x1, y1 in screen-capture pixels
MARKER_R = 27


def paint_out(rgb, mask):
    """Fill the masked pixels with the value of the nearest unmasked pixel."""
    idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    return rgb[tuple(idx)]


def to_x(px):
    return X0_VAL + (px - X0_PX) / X_SCALE


def to_y(px):
    return Y0_VAL + (px - Y0_PX) / Y_SCALE


def main():
    shot = np.array(Image.open(SHOT).convert('RGB'))
    mask = np.zeros(shot.shape[:2], bool)
    x0, y0, x1, y1 = CALLOUT
    mask[y0:y1, x0:x1] = True
    yy, xx = np.mgrid[:shot.shape[0], :shot.shape[1]]
    for mx, my in MARKERS_PX.values():
        mask |= (xx - mx) ** 2 + (yy - my) ** 2 <= MARKER_R ** 2
    shot = paint_out(shot, mask)
    photo = shot[FRAME['top'] + 1:FRAME['bottom'],
                 FRAME['left'] + 1:FRAME['right']]

    # colour key resampled from the original, top (head) to bottom (tail tip)
    strip = shot[FRAME['top'] + 2:FRAME['bottom'] - 1,
                 CBAR[0] + 6:CBAR[1] - 6].mean(1) / 255.0
    cmap = ListedColormap(strip[::-1])

    rgb = np.array(Image.open(DRAW).convert('RGB')).astype(float)
    alpha = np.array(Image.open(DRAW_A).convert('L'))
    rgb[alpha < 250] = 255.0          # avoid a dark halo around the drawing
    rgb = rgb.astype(np.uint8)
    ys, xs = np.where(alpha > 10)
    draw = np.dstack([rgb, alpha])[ys.min():ys.max() + 1,
                                   xs.min():xs.max() + 1]

    ext = [to_x(FRAME['left']), to_x(FRAME['right']),
           to_y(FRAME['bottom']), to_y(FRAME['top'])]

    FW = 7.1
    PW = 4.05 * R
    PH = PW * (ext[3] - ext[2]) / (ext[1] - ext[0]) * -1
    PH = abs(PW * (to_y(FRAME['bottom']) - to_y(FRAME['top'])) /
             (ext[1] - ext[0]))
    TOP, BOT = 0.30, 0.42
    FH = TOP + PH + BOT
    FW = 0.52 + PW + 0.16 + 0.15 * R + 0.62 + PH * draw.shape[1] / draw.shape[0] + 0.05
    fig = plt.figure(figsize=(FW, FH))

    def place(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    ax = place(0.52, BOT, PW, PH)
    ax.imshow(photo, extent=ext, aspect='auto', interpolation='lanczos',
              origin='upper')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])          # y increases downwards, as in the original
    ax.set_xticks(np.arange(200, 290, 10))
    ax.set_yticks(np.arange(120, 190, 10))
    ax.set_xlabel('$X$ position (px)')
    ax.set_ylabel('$Y$ position (px)')
    ax.set_title('$N_{\\mathrm{markers}}=3$', fontsize=9 * K)

    for nm, (mx, my) in MARKERS_PX.items():
        col = cmap({'Head': 1.0, 'Tail_B': 0.5, 'Tail_E': 0.0}[nm])
        ax.scatter(to_x(mx), to_y(my), s=150, color=col, ec='k', lw=0.6,
                   zorder=4)
    ax.annotate('Tail_E', xy=(to_x(651) + 1.2, to_y(297) - 1.2),
                xytext=(to_x(790), to_y(240)), fontsize=8 * K, ha='left',
                va='center',
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.4',
                          lw=0.6),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=0.7,
                                shrinkA=1, shrinkB=1))

    # ---- colour key
    cax = place(0.52 + PW + 0.16, BOT, 0.15 * R, PH)
    cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                      cax=cax)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Tail_E', 'Tail_B', 'Head'])
    cb.ax.tick_params(labelsize=8 * K, length=2, pad=2)
    for lab, v in zip(cb.ax.get_yticklabels(), (0.0, 0.5, 1.0)):
        lab.set_color(cmap(v))          # each label in the colour of its marker
    cb.outline.set_linewidth(0.6)

    # ---- schematic drawing with the three landmarks
    dw = PH * draw.shape[1] / draw.shape[0]
    axd = place(0.52 + PW + 0.16 + 0.15 * R + 0.62, BOT, dw, PH)
    axd.imshow(draw, interpolation='lanczos')
    axd.set_axis_off()
    h, w = draw.shape[:2]
    for nm, (fx, fy) in DRAW_DOTS.items():
        col = cmap({'Head': 1.0, 'Tail_B': 0.5, 'Tail_E': 0.0}[nm])
        axd.scatter(fx * w, fy * h, s=110, color=col, ec='k', lw=0.6,
                    zorder=3, clip_on=False)

    fig.savefig('fig_dlc_labels.pdf')
    plt.close(fig)
    print('ok fig_dlc_labels.pdf | x %.1f-%.1f  y %.1f-%.1f | photo %s'
          % (ext[0], ext[1], ext[3], ext[2], photo.shape[:2]))


if __name__ == '__main__':
    main()
