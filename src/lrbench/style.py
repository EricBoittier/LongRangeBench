"""Figure style: J. Phys. Chem.-like, Tufte data-ink, 89/183 mm widths,
6-8 pt sans, greys for context, Okabe-Ito accents, direct labels, takeaway titles."""
import matplotlib as mpl
MM = 1 / 25.4; W1, W2 = 89 * MM, 183 * MM
INK = "#1a1a1a"; G1, G2, G3 = "#4d4d4d", "#8c8c8c", "#c7c7c7"
ACC = "#0072B2"; ACC2 = "#D55E00"; ACC3 = "#009E73"
ROLE = {"DFT": INK, "mp": G2, "Z": G3, "Ectl": G3, "Fa": ACC, "FQ": ACC2, "oracle": ACC3}
NAME = {"Z": "zero-shot", "Ectl": "Ectl", "E": "E", "Fa": "Fa", "Fb": "Fb", "FQ": "FQ", "FQL": "FQL", "S1": "S1", "SFa": "SFa", "damp4": "damp4", "T": "T", "Kfix": "Kfix",
        "G4b": "G4b", "Gs4": "Gs4", "Fa0": "Fa, gate 0", "Fb0": "Fb, gate 0", "Fa0+q_formal": "oracle: + exact Coulomb, formal q", "Fa0+q_Löwdin": "oracle: + exact Coulomb, Löwdin q"}
def setup():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.titleweight": "bold", "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.edgecolor": G1, "xtick.color": G1, "ytick.color": G1,
        "axes.labelcolor": INK, "text.color": INK, "lines.linewidth": 1.0, "lines.markersize": 3, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600, "figure.dpi": 150, "axes.titlepad": 4})
def letter(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
def title(ax, s):  # one-line takeaway, left aligned
    ax.set_title(s, loc="left", fontsize=6.6, fontweight="bold")
def note(ax, s, x=0.0, y=-0.30, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=5.3, color=G1, va="top", ha="left", **kw)
def endlabel(ax, x, y, s, color, dx=3, dy=0, **kw):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, color=color, va="center", **kw)
def save(fig, name, out):
    for ext in ("pdf", "svg", "png"): fig.savefig(f"{out}/{name}.{ext}", bbox_inches="tight", facecolor="white")
PALETTE = [ACC, ACC2, ACC3, "#CC79A7", "#E69F00", "#56B4E9", G1, G2]
