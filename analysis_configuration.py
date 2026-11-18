# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths ----
ROOT = os.path.abspath(os.environ.get("KIWI_ROOT", os.path.dirname(__file__)))
AMP = os.path.abspath(os.environ.get("KIWI_AMPLICON_DIR", os.path.join(ROOT, "amplicon_results")))
ENV_DIR = os.path.abspath(os.environ.get("KIWI_ASV_TABLE_DIR", os.path.join(AMP, "asv_tables")))
RESULT = os.path.join(ROOT, "result")
FIGS = os.path.join(RESULT, "figs")
TABLES = os.path.join(RESULT, "tables")
DATA = os.path.join(RESULT, "data")
R_IN = os.path.join(RESULT, "r_in")
R_OUT = os.path.join(RESULT, "r_out")
TEXT = os.path.join(RESULT, "text")

META_PATH = os.path.join(R_IN, "sample_meta.tsv")

RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

# ----------------------------------------------------------------- order ----
COMP_ORDER = ["BS", "RS", "R", "S", "L", "F"]
COMP_LABEL = {"BS": "Bulk soil", "RS": "Rhizosphere soil", "R": "Root",
              "S": "Stem", "L": "Leaf", "F": "Fruit"}
COMP_LABELS = [COMP_LABEL[c] for c in COMP_ORDER]
REGIONS = ["WH", "SH", "SX"]
REG_MAP = {"MX": "SX", "BZ": "SH"}
COLORS = {"WH": "#3C5488", "SH": "#E64B35", "SX": "#00A087"}
AAAS = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F",
        "#8491B4", "#91D1C2", "#7E6148", "#DC0000", "#B09C85"]
STACK_COLORS = ["#A0C8E0", "#1878B0", "#B0D888", "#30A028",
             "#F89898", "#E01818", "#F8B868", "#F87800"]
CI_GRAY = "#7F7F7F"
OTHER_GRAY = "#B09C85"

DATASETS = {"16S": "Bacteria", "ITS": "Fungi"}

# ------------------------------------------------------- matplotlib style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel(ax, letter, x=-0.22, y=1.08, fontsize=9):
    """Add a lowercase label to the upper left of a plot panel."""
    ax.text(x, y, "%s)" % str(letter).lower(), transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top",
            family="serif")


def frame_all(ax, lw=0.8):
    """Display all four axes borders."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
