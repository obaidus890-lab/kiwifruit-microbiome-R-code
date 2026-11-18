# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

warnings.simplefilter("ignore")

import analysis_helpers
from analysis_configuration import COMP_ORDER, COMP_LABEL, FIGS, TABLES
from analyze_fruit_metabolome_modules_and_source_tracking import module_eigengenes
from analyze_fruit_metabolome_microbiome_associations import sample_map, MET_TO_REG

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 5.2,
    "legend.fontsize": 6,
    "axes.unicode_minus": False,
})

CHAIN = ["BS", "RS", "R", "S", "L", "F"]   # Analysis note: see README and original laboratory scripts for methodological details.
ADJ = [("BS", "RS"), ("RS", "R"), ("R", "S"), ("S", "L"), ("L", "F")]
# Analysis note: see README and original laboratory scripts for methodological details.
INDIRECT = {
    "L":  [("L", "F")],
    "S":  [("S", "L"), ("L", "F")],
    "R":  [("R", "S"), ("S", "L"), ("L", "F")],
    "RS": [("RS", "R"), ("R", "S"), ("S", "L"), ("L", "F")],
    "BS": [("BS", "RS"), ("RS", "R"), ("R", "S"), ("S", "L"), ("L", "F")],
}


def panel(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def bh_q(pvals):
    return analysis_helpers.bh_q(pvals)


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def mepink_taxa():
    fg = pd.read_csv(os.path.join(TABLES, "tab_sec16_fruit_genus_module.tsv"),
                     sep="\t")
    mp = fg[(fg["module"] == "MEpink") & (fg["FDR"] < 0.05)].copy()
    mp = mp.sort_values("rho").reset_index(drop=True)
    mp["kingdom"] = mp["dataset"].map({"16S": "bacteria", "ITS": "fungi"})
    out = mp[["kingdom", "genus", "rho", "p", "FDR"]]
    out.to_csv(os.path.join(TABLES, "tab_sec16_mepink_taxa.tsv"),
               sep="\t", index=False)
    return out


def genus_compartment_loads(genera):
    """Research analysis helper. See README for pipeline and methodology."""
    dat = analysis_helpers.load_dataset("ITS")
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")  # genus x sample
    _, _, fmap = sample_map()  # met_sample -> mic_F_sample
    pref2reg = {"WH": "WH", "BZ": "SH", "SX": "SX"}

    loads = {}
    for g in genera:
        if g not in G.index:
            continue
        row = G.loc[g]
        mat = {}
        for met_s, mic_s in fmap.items():
            parts = mic_s.split("_")
            pref, rep = parts[0], parts[2]
            vec = {}
            for c in COMP_ORDER:
                sid = "%s_%s_%s" % (pref, c, rep)
                vec[c] = float(row.get(sid, 0.0))
            mat[met_s] = vec
        loads[g] = pd.DataFrame(mat).T  # met_sample x COMP_ORDER
    return loads


def load_heatmap_matrix(genera, loads):
    'genus x COMP_ORDER \u5747\u503c log10 \u8f7d\u91cf\uff0827 \u6837\u672c\u5747\u503c\uff09\u3002'
    rows = []
    for g in genera:
        if g not in loads:
            continue
        m = loads[g]
        rows.append({"genus": g, **{c: np.log10(m[c].replace(0, np.nan).mean() + 1)
                                    for c in COMP_ORDER}})
    df = pd.DataFrame(rows).set_index("genus")
    return df[COMP_ORDER]


def compartment_spearman(genera, loads, eig):
    '\u6bcf\u4e2a\u5c5e \xd7 \u6bcf\u4e2a compartment\uff1a\u8f7d\u91cf vs \u679c\u5b9e MEpink Spearman\u3002'
    me = eig["MEpink"]
    rows = []
    for g in genera:
        if g not in loads:
            continue
        m = loads[g]
        common_idx = m.index.intersection(me.index)
        for c in COMP_ORDER:
            x = np.log10(m.loc[common_idx, c].replace(0, np.nan).fillna(0).values + 1)
            y = me.loc[common_idx].values
            r, p = spearmanr(x, y)
            rows.append({"genus": g, "compartment": c, "n": len(common_idx),
                         "rho": r, "p": p})
    df = pd.DataFrame(rows)
    df["FDR"] = bh_q(df["p"].values)
    return df


def plot_tracking(genera, hm, sp, mepink_rho):
    '\u53cc\u9762\u677f\uff1a\u5de6 = \u8f7d\u91cf\u68af\u5ea6\uff08\u884c Z\uff09\uff0c\u53f3 = \u03c1 vs MEpink\u3002'
    # Analysis note: see README and original laboratory scripts for methodological details.
    argmax = hm.apply(lambda r: CHAIN.index(COMP_ORDER[np.argmax(r.values)]), axis=1)
    order = sorted(hm.index, key=lambda g: (argmax[g], -hm.loc[g].mean()))
    hm = hm.loc[order]
    sp_mat = sp.pivot(index="genus", columns="compartment", values="rho").loc[order, COMP_ORDER]
    fdr_mat = sp.pivot(index="genus", columns="compartment", values="FDR").loc[order, COMP_ORDER]

    # Analysis note: see README and original laboratory scripts for methodological details.
    M = hm.values
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1],
                                                "wspace": 0.42})
    # ---- panel a: load heatmap (row Z) ----
    im1 = ax1.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax1.set_xticks(range(len(COMP_ORDER)))
    ax1.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER], rotation=40, ha="right")
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order, fontstyle="italic")
    ax1.set_title("Mean log₁₀(copies/g+1)  [row Z-score]", fontsize=7.5)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("row Z-score", size=6.5)
    cb1.ax.tick_params(labelsize=5.5)
    panel(ax1, "a")

    # ---- panel b: Spearman rho vs MEpink ----
    im2 = ax2.imshow(sp_mat.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(COMP_ORDER)))
    ax2.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER], rotation=40, ha="right")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([""] * len(order))
    ax2.set_title("Spearman ρ (compartment load vs fruit MEpink)", fontsize=7.5)
    # Analysis note: see README and original laboratory scripts for methodological details.
    for i in range(sp_mat.shape[0]):
        for j in range(sp_mat.shape[1]):
            f = fdr_mat.values[i, j]
            star = "***" if f < 0.001 else "**" if f < 0.01 else "*" if f < 0.05 else ""
            if star:
                ax2.text(j, i, star, ha="center", va="center", fontsize=5.5,
                         color="black")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("Spearman ρ", size=6.5)
    cb2.ax.tick_params(labelsize=5.5)
    panel(ax2, "b")

    fig.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.10)
    return fig


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def feast_indirect_contrib(ds):
    '\u8fd4\u56de DataFrame: met_sample x [BS,RS,R,S,L] \u95f4\u63a5\u2192F \u8d21\u732e\u7387\u3002'
    f = pd.read_csv(os.path.join(TABLES, "tab_eco_feast_%s.tsv" % ds), sep="\t")
    _, _, fmap = sample_map()
    inv = {v: k for k, v in fmap.items()}  # mic_F_sample -> met_sample

    # (from,to,sink_sample) -> contribution
    lookup = {(r["from"], r["to"], r["sink_sample"]): r["contribution"]
              for _, r in f.iterrows()}

    rows = {}
    for met_s, mic_s in fmap.items():
        parts = mic_s.split("_")
        pref, rep = parts[0], parts[2]
        # Analysis note: see README and original laboratory scripts for methodological details.
        step_c = {}
        for (frm, to) in ADJ:
            sink = "%s_%s_%s" % (pref, to, rep)
            step_c[(frm, to)] = lookup.get((frm, to, sink), np.nan)
        # Analysis note: see README and original laboratory scripts for methodological details.
        contrib = {}
        for src, steps in INDIRECT.items():
            val = 1.0
            for s in steps:
                val *= step_c.get(s, np.nan)
            contrib[src] = val
        rows[met_s] = contrib
    return pd.DataFrame(rows).T  # met_sample x [BS,RS,R,S,L]


def feast_module_corr(ds, eig):
    '5 \u4e0a\u6e38 \xd7 {MEpink, MEturquoise} Spearman\uff0cBH\u3002'
    contrib = feast_indirect_contrib(ds)
    modules = ["MEpink", "MEturquoise"]
    rows = []
    for src in ["BS", "RS", "R", "S", "L"]:
        for me in modules:
            idx = contrib.index.intersection(eig.index)
            x = contrib.loc[idx, src].replace(np.nan, 0.0).values
            y = eig.loc[idx, me].values
            r, p = spearmanr(x, y)
            rows.append({"dataset": ds, "source": src, "module": me,
                         "n": len(idx), "rho": r, "p": p,
                         "method": "chain_product"})
    df = pd.DataFrame(rows)
    df["FDR"] = bh_q(df["p"].values)
    return df


# ------------------------------------------------------------------ #
def main():
    eig, _, _ = module_eigengenes()
    print("eigengene modules:", list(eig.columns))

    # Analysis note: see README and original laboratory scripts for methodological details.
    taxa = mepink_taxa()
    print('\n[\u5206\u67901] MEpink FDR<0.05 taxa: %d (%s)' %
          (len(taxa), taxa["kingdom"].value_counts().to_dict()))
    genera = taxa["genus"].tolist()
    loads = genus_compartment_loads(genera)
    hm = load_heatmap_matrix(genera, loads)
    hm_out = hm.copy()
    hm_out["MEpink_rho"] = taxa.set_index("genus")["rho"]
    hm_out.to_csv(os.path.join(TABLES, "tab_sec16_mepink_load_heatmap.tsv"),
                  sep="\t")
    sp = compartment_spearman(genera, loads, eig)
    sp.to_csv(os.path.join(TABLES, "tab_sec16_mepink_compartment_spearman.tsv"),
              sep="\t", index=False)

    fig = plot_tracking(genera, hm, sp, taxa.set_index("genus")["rho"])
    out_fig = os.path.join(FIGS, "fig_sec16_mepink_tracking.png")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print("  ->", out_fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    sig_sp = sp[sp["FDR"] < 0.05].sort_values("rho")
    print('\n[\u5206\u67901 step4] compartment load vs MEpink \u663e\u8457 (FDR<0.05): %d / %d'
          % (len(sig_sp), len(sp)))
    print(sig_sp.head(12).round(3).to_string(index=False))
    # Analysis note: see README and original laboratory scripts for methodological details.
    cnt = sp[sp["FDR"] < 0.05].groupby("compartment").size().reindex(COMP_ORDER).fillna(0).astype(int)
    print('  \u5404 compartment \u663e\u8457\u5c5e\u6570:', cnt.to_dict())
    # Analysis note: see README and original laboratory scripts for methodological details.
    chk = sp[sp["compartment"] == "F"].set_index("genus")["rho"]
    orig = taxa.set_index("genus")["rho"]
    print('  \u6821\u9a8c F \u8f7d\u91cf vs MEpink \u03c1 \u4e0e\u539f\u59cb MEpink \u5173\u8054 \u03c1 \u4e00\u81f4\u6027:',
          "OK" if np.allclose(chk.reindex(orig.index), orig, atol=0.02) else "MISMATCH")

    # Analysis note: see README and original laboratory scripts for methodological details.
    frames = []
    for ds in ["16S", "ITS"]:
        d = feast_module_corr(ds, eig)
        frames.append(d)
    feast_df = pd.concat(frames, ignore_index=True)
    feast_df.to_csv(os.path.join(TABLES, "tab_sec16_feast_mepink.tsv"),
                    sep="\t", index=False)
    print('\n[\u5206\u67902] FEAST \u95f4\u63a5\u4e0a\u6e38\u2192F \u8d21\u732e \xd7 \u5173\u952e\u6a21\u5757 (chain product):')
    print(feast_df.round(4).to_string(index=False))
    sig_f = feast_df[feast_df["FDR"] < 0.05]
    print('\n  \u663e\u8457 (FDR<0.05): %d / %d' % (len(sig_f), len(feast_df)))
    if len(sig_f):
        print(sig_f.round(4).to_string(index=False))

    # Analysis note: see README and original laboratory scripts for methodological details.
    ic = feast_indirect_contrib("ITS")
    print('\n[\u5206\u67902] ITS \u95f4\u63a5\u4e0a\u6e38\u2192F \u8d21\u732e\u5206\u5e03 (median):')
    print(ic.median().round(3).to_dict())


if __name__ == "__main__":
    sys.exit(main())
