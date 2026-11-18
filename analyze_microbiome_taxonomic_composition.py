# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analysis_configuration as C
import analysis_helpers as U

TOP_PHYLA = 8          # Analysis note: see README and original laboratory scripts for methodological details.
TOP_GENUS_HEAT = 10    # Analysis note: see README and original laboratory scripts for methodological details.
N_REPR_GENUS = 8       # Analysis note: see README and original laboratory scripts for methodological details.


# ----------------------------------------------------------------- prep ----
def prepare():
    '\u8f7d\u5165\u4e24\u4e2a\u6570\u636e\u96c6\uff0c\u8fd4\u56de\u6bcf\u4e2a\u6570\u636e\u96c6\u7684\u6837\u672c\u7ea7\u805a\u5408\u8868\u3002'
    out = {}
    loads = []
    for ds in ("16S", "ITS"):
        D = U.load_dataset(ds)
        abss, rel, tax, meta = D["asv_abs"], D["asv_rel"], D["tax"], D["meta"]
        samples = [s for s in meta.index if s in abss.columns]

        abs_smp = abss[samples].T.astype(float)          # Analysis note: see README and original laboratory scripts for methodological details.
        rel_smp = rel[samples].T.astype(float)           # Analysis note: see README and original laboratory scripts for methodological details.
        m = meta.loc[samples]

        phy_abs = U.agg_tax(abs_smp.T, tax, "phylum").T   # Analysis note: see README and original laboratory scripts for methodological details.
        gen_abs = U.agg_tax(abs_smp.T, tax, "genus").T
        gen_abs = gen_abs.loc[:, gen_abs.sum(axis=0) > 0]
        phy_rel = U.agg_tax(rel_smp.T, tax, "phylum").T
        gen_rel = U.agg_tax(rel_smp.T, tax, "genus").T
        gen_rel = gen_rel.loc[:, gen_rel.sum(axis=0) > 0]
        # Analysis note: see README and original laboratory scripts for methodological details.
        phy_rel = phy_rel.div(phy_rel.sum(axis=1), axis=0)
        gen_rel = gen_rel.div(gen_rel.sum(axis=1), axis=0)

        out[ds] = {"samples": samples, "meta": m,
                   "phy_abs": phy_abs, "gen_abs": gen_abs,
                   "phy_rel": phy_rel, "gen_rel": gen_rel}

        tot = abs_smp.sum(axis=1)
        tmp = pd.DataFrame({"sample_id": samples, "dataset": ds,
                            "total_abs_copies": tot.values,
                            "log10_load": np.log10(tot.values)})
        loads.append(tmp)
    return out, pd.concat(loads, ignore_index=True)


def write_outputs(O, loads):
    for ds in ("16S", "ITS"):
        d = O[ds]
        for level, lab in (("gen", "genus"), ("phy", "phylum")):
            for kind in ("abs", "rel"):
                df = d["%s_%s" % (level, kind)]
                df.index.name = "sample_id"
                df.to_csv(os.path.join(C.DATA, "%s_%s_%s.tsv" % (lab, kind, ds)),
                          sep="\t", float_format="%.6g")
        # Analysis note: see README and original laboratory scripts for methodological details.
        g = d["gen_abs"].copy()
        g.index.name = "sample_id"
        g.to_csv(os.path.join(C.R_IN, "genus_abs_%s.tsv" % ds),
                 sep="\t", float_format="%.6g")

    loads.to_csv(os.path.join(C.DATA, "sample_loads.tsv"), sep="\t",
                 index=False, float_format="%.6g")

    meta16 = O["16S"]["meta"]
    sm = pd.DataFrame({"sample_id": meta16.index,
                       "compartment": meta16["compartment"].values,
                       "region": meta16["region"].values,
                       "replicate": meta16["replicate"].values,
                       "underground": meta16["underground"].values,
                       "plant": meta16["plant"].values})
    sm.to_csv(os.path.join(C.R_IN, "sample_meta.tsv"), sep="\t", index=False)


# ----------------------------------------------------- fig 12a/b phylum ----
def fig_phylum(ds, letter0, fname):
    d = O[ds]
    phy_rel, phy_abs = d["phy_rel"], d["phy_abs"]
    comp = d["meta"]["compartment"]

    med_rel = phy_rel.groupby(comp).median().loc[C.COMP_ORDER]      # 6 x P
    med_abs = phy_abs.groupby(comp).median().loc[C.COMP_ORDER]
    top = med_rel.mean(axis=0).sort_values(ascending=False).index[:TOP_PHYLA]
    # Analysis note: see README and original laboratory scripts for methodological details.
    other_vals = np.clip(1.0 - med_rel[top].sum(axis=1).values, 0.0, None)

    fig = plt.figure(figsize=(8.6, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Analysis note: see README and original laboratory scripts for methodological details.
    x = np.arange(len(C.COMP_ORDER))
    bottom = np.zeros(len(x))
    cols = list(top) + ["Other"]
    for i, ph in enumerate(cols):
        vals = med_rel[ph].values if ph != "Other" else other_vals
        color = C.STACK_COLORS[i] if ph != "Other" else C.OTHER_GRAY
        ax1.bar(x, vals, bottom=bottom, color=color, width=0.85,
                edgecolor="white", linewidth=0.4, label=ph)
        bottom += vals
    U.set_xticks_labels(ax1, x, C.COMP_LABELS, rotation=30, ha="right")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Relative abundance (median)")
    ax1.set_xlabel("Compartment")
    ax1.set_title("%s relative composition" % C.DATASETS[ds])
    C.panel(ax1, letter0, x=-0.28)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for i, ph in enumerate(top):
        v = med_abs[ph].values.astype(float)
        v[v <= 0] = np.nan
        v = np.log10(v)
        ax2.plot(x, v, "-o", color=C.STACK_COLORS[i], markersize=2.6,
                 linewidth=1.6, label=ph)
    U.set_xticks_labels(ax2, x, C.COMP_LABELS, rotation=30, ha="right")
    ax2.set_xlabel("Compartment")
    ax2.set_ylabel("Absolute abundance\n(log10 median copies)")
    ax2.set_title("%s absolute copy number" % C.DATASETS[ds])
    # Analysis note: see README and original laboratory scripts for methodological details.
    handles = [Line2D([0], [0], color=C.STACK_COLORS[i], lw=1.6, marker="o",
                      markersize=2.6, label=ph) for i, ph in enumerate(top)]
    handles.append(Patch(facecolor=C.OTHER_GRAY, edgecolor="white",
                         linewidth=0.4, label="Other"))
    ax2.legend(handles=handles, fontsize=7, frameon=False,
               loc="center left", bbox_to_anchor=(1.02, 0.5),
               handlelength=1.5, labelspacing=0.45)
    C.panel(ax2, chr(ord(letter0) + 1), x=-0.18)

    fig.savefig(os.path.join(C.FIGS, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ fig 12c genus heatmap ----
def heat_panel(ax, gen_rel, comp, title, letter):
    med = gen_rel.groupby(comp).median().loc[C.COMP_ORDER]   # 6 x G
    overall = med.mean(axis=0)
    top = overall.sort_values(ascending=False).index[:TOP_GENUS_HEAT]
    mat = med[top].T                                         # G x 6
    mat = mat.div(mat.sum(axis=1), axis=0)                   # Analysis note: see README and original laboratory scripts for methodological details.

    im = ax.imshow(mat.values, aspect="auto", cmap="Reds")
    for i, g in enumerate(top):
        j = int(np.argmax(mat.values[i]))
        ax.text(j, i, "*", ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
    U.set_xticks_labels(ax, np.arange(len(C.COMP_ORDER)), C.COMP_LABELS,
                        rotation=30, ha="right")
    U.set_yticks_labels(ax, np.arange(len(top)), top)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xticks(np.arange(len(C.COMP_ORDER)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(top)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlim(-0.5, len(C.COMP_ORDER) - 0.5)   # Analysis note: see README and original laboratory scripts for methodological details.
    ax.tick_params(which="both", length=0)
    C.frame_all(ax)
    # Analysis note: see README and original laboratory scripts for methodological details.
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    C.panel(ax, letter, x=-0.55)


def fig_genus_specificity(fname):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 8.2))
    heat_panel(axes[0], O["16S"]["gen_rel"], O["16S"]["meta"]["compartment"],
               "Bacteria (16S)", "a")
    heat_panel(axes[1], O["ITS"]["gen_rel"], O["ITS"]["meta"]["compartment"],
               "Fungi (ITS)", "b")
    axes[1].set_xlabel("Compartment")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------- fig 12d abs vs rel trajectories
def select_genera(gen_abs, gen_rel, comp):
    '\u9009\u4ee3\u8868\u6027\u5c5e\uff1a\u9ad8\u7edd\u5bf9\u4e30\u5ea6 + abs/rel \u8f68\u8ff9\u5206\u6b67\u5927\u8005\uff0c\u53bb\u91cd\u81f3 N_REPR_GENUS\u3002'
    med_abs = gen_abs.groupby(comp).median().loc[C.COMP_ORDER]
    med_rel = gen_rel.groupby(comp).median().loc[C.COMP_ORDER]
    overall_abs = med_abs.mean(axis=0)
    pool = overall_abs.sort_values(ascending=False).index[:30]

    def norm(v):
        v = v.astype(float)
        m = np.median(v)
        return v / m if m > 0 else v / v.max() if v.max() > 0 else v

    div = {}
    for g in pool:
        na, nr = norm(med_abs[g].values), norm(med_rel[g].values)
        div[g] = float(np.mean(np.abs(na - nr)))
    div_s = pd.Series(div).sort_values(ascending=False)
    sel = list(div_s.index[: N_REPR_GENUS // 2])
    for g in overall_abs.sort_values(ascending=False).index:
        if g not in sel:
            sel.append(g)
        if len(sel) >= N_REPR_GENUS:
            break
    return sel, med_abs, med_rel


def fig_abs_vs_rel(fname):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = np.arange(len(C.COMP_ORDER))
    for ax, ds, letter in zip(axes, ("16S", "ITS"), ("a", "b")):
        d = O[ds]
        sel, med_abs, med_rel = select_genera(
            d["gen_abs"], d["gen_rel"], d["meta"]["compartment"])

        def norm(v):
            v = v.astype(float)
            m = np.median(v)
            return v / m if m > 0 else v / v.max() if v.max() > 0 else v

        for i, g in enumerate(sel):
            c = C.AAAS[i % 10]
            na = norm(med_abs[g].values)
            nr = norm(med_rel[g].values)
            ax.plot(x, na, "-o", color=c, markersize=2.7, linewidth=1.4,
                    label=g)
            ax.plot(x, nr, "--s", color=c, markersize=2.3, linewidth=1.2,
                    alpha=0.75)
        U.set_xticks_labels(ax, x, C.COMP_LABELS, rotation=30, ha="right")
        ax.set_title("%s (%s)" % (C.DATASETS[ds], ds))
        ax.set_ylabel("Median-normalized abundance")
        ax.legend(fontsize=6.5, frameon=False, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), handlelength=1.8,
                  labelspacing=0.45)
        C.panel(ax, letter, x=-0.16)
    axes[0].set_xlabel("Compartment")
    axes[1].set_xlabel("Compartment")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- tables ----
def table_dominant(fname):
    rows = []
    for ds in ("16S", "ITS"):
        d = O[ds]
        gen_abs, m = d["gen_abs"], d["meta"]
        med_all = gen_abs.groupby(m["compartment"]).median()
        med_rr = gen_abs.groupby([m["region"], m["compartment"]]).median()
        for comp in C.COMP_ORDER:
            top5 = med_all.loc[comp].sort_values(ascending=False).head(5)
            for rank, (g, v) in enumerate(top5.items(), 1):
                in_all = []
                for reg in C.REGIONS:
                    sub = med_rr.loc[reg].loc[comp] if reg in med_rr.index else None
                    if sub is not None and g in sub.sort_values(
                            ascending=False).head(5).index:
                        in_all.append(reg)
                rows.append({"dataset": ds, "compartment": comp, "rank": rank,
                             "genus": g, "median_abs": v,
                             "top_in_regions": ";".join(in_all),
                             "consistent_all_regions":
                                 "yes" if len(in_all) == 3 else "no"})
    pd.DataFrame(rows).to_csv(os.path.join(C.TABLES, fname), sep="\t",
                              index=False, float_format="%.6g")


def table_specificity(fname):
    rows = []
    for ds in ("16S", "ITS"):
        d = O[ds]
        gen_abs, gen_rel, m = d["gen_abs"], d["gen_rel"], d["meta"]
        comp = m["compartment"]
        med_rel = gen_rel.groupby(comp).median().loc[C.COMP_ORDER]
        med_abs = gen_abs.groupby(comp).median().loc[C.COMP_ORDER]
        keep = med_rel.mean(axis=0)[med_rel.mean(axis=0) >= 1e-3].index
        norm = med_rel.div(med_rel.sum(axis=0), axis=1)   # Analysis note: see README and original laboratory scripts for methodological details.
        for g in keep:
            ec = norm[g].idxmax()
            rows.append({"dataset": ds, "genus": g,
                         "enriched_compartment": ec,
                         "specificity": norm.loc[ec, g],
                         "median_abs_overall": med_abs[g].median(),
                         "median_rel_overall": med_rel[g].median()})
    tab = pd.DataFrame(rows).sort_values(
        ["dataset", "median_abs_overall"], ascending=[True, False])
    tab.to_csv(os.path.join(C.TABLES, fname), sep="\t", index=False,
               float_format="%.6g")


# ----------------------------------------------------------------- main ----
if __name__ == "__main__":
    for dd in (C.DATA, C.FIGS, C.TABLES, C.R_IN):
        os.makedirs(dd, exist_ok=True)

    O, loads = prepare()
    write_outputs(O, loads)
    print("[data] aggregation tables + sample_loads + r_in written")

    fig_phylum("16S", "a", "fig_12a_bacteria_phylum.png")
    fig_phylum("ITS", "a", "fig_12b_fungi_phylum.png")
    fig_genus_specificity("fig_12c_genus_specificity.png")
    fig_abs_vs_rel("fig_12d_abs_vs_rel.png")
    print("[figs] 4 figures written")

    table_dominant("tab_12_dominant_genera.tsv")
    table_specificity("tab_12_genus_specificity.tsv")
    print("[tables] 2 tables written")
    sys.exit(0)
