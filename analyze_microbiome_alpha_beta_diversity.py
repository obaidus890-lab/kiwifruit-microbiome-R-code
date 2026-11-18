# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import t as tdist

import analysis_configuration
from analysis_configuration import (AMP, COMP_ORDER, COMP_LABEL, REGIONS, REG_MAP, COLORS,
                    DATASETS, FIGS, TABLES, DATA, R_IN, panel)
from analysis_helpers import bc_distance, bh_q, sig_label, set_xticks_labels
from analysis_configuration import CI_GRAY as C_CI

ALPHA_DIR = DATA          # Analysis note: see README and original laboratory scripts for methodological details.
PERM_PATH = os.path.join(TABLES, "permanova_multifactor.tsv")
META_IN = os.path.join(R_IN, "sample_meta.tsv")

ADJ_PAIRS = list(zip(COMP_ORDER, COMP_ORDER[1:]))           # BS-RS ... L-F
BELOW = ["BS", "RS", "R"]
ABOVE = ["S", "L", "F"]
TERM_LABEL = {"compartment": "Compartment", "region": "Region",
              "compartment:region": "Interaction"}
TERM_COLOR = {"compartment": "#3C5488", "region": "#E64B35",
              "compartment:region": "#00A087"}
RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ load ----
def load_alpha(ds):
    p = os.path.join(ALPHA_DIR, "alpha9_%s.tsv" % ds)
    df = pd.read_csv(p, sep="\t")
    df = df.dropna(subset=["sample_id"])
    return df[["sample_id", "Shannon", "region", "compartment"]].copy()


def load_genus_abs(ds):
    '\u6837\u672c\xd7\u5c5e\u7edd\u5bf9\u62f7\u8d1d\u8868 + \u5730\u533a/\u90e8\u4f4d\u6ce8\u91ca\uff08\u7ecf r_in/sample_meta.tsv\uff09\u3002'
    p = os.path.join(DATA, "genus_abs_%s.tsv" % ds)
    df = pd.read_csv(p, sep="\t", index_col=0)
    meta = pd.read_csv(META_IN, sep="\t").set_index("sample_id")
    miss = [s for s in df.index if s not in meta.index]
    if miss:
        raise RuntimeError('genus_abs \u6837\u672c\u7f3a meta \u6ce8\u91ca: %s' % miss[:5])
    ann = meta.loc[df.index, ["compartment", "region"]]
    return df, ann


# ----------------------------------------------------------------- stats ----
def alpha_tests(alphas):
    '\u76f8\u90bb\u5bf9 MW\uff08\u5168\u6837\u672c+\u5206\u5730\u533a\uff09+ compartment \u5185\u5730\u533a\u95f4 KW\uff1bBH \u6821\u6b63\u3002'
    rows = []
    for ds, df in alphas.items():
        # Analysis note: see README and original laboratory scripts for methodological details.
        for a, b in ADJ_PAIRS:
            x = df.loc[df["compartment"] == a, "Shannon"].values
            y = df.loc[df["compartment"] == b, "Shannon"].values
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(dict(test="MW", dataset=ds, scope="overall",
                             region="ALL", comparison="%s-%s" % (a, b),
                             n=len(x) + len(y), stat=u, p=p))
        # Analysis note: see README and original laboratory scripts for methodological details.
        for reg in REGIONS:
            sub = df[df["region"] == reg]
            for a, b in ADJ_PAIRS:
                x = sub.loc[sub["compartment"] == a, "Shannon"].values
                y = sub.loc[sub["compartment"] == b, "Shannon"].values
                if len(x) < 2 or len(y) < 2:
                    continue
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                rows.append(dict(test="MW", dataset=ds, scope="regional",
                                 region=reg, comparison="%s-%s" % (a, b),
                                 n=len(x) + len(y), stat=u, p=p))
        # Analysis note: see README and original laboratory scripts for methodological details.
        for comp in COMP_ORDER:
            groups = [g["Shannon"].values for _, g in
                      df[df["compartment"] == comp].groupby("region", sort=False)
                      if len(g) >= 2]
            if len(groups) < 2:
                continue
            h, p = kruskal(*groups)
            rows.append(dict(test="KW", dataset=ds, scope="overall",
                             region="ALL", comparison=comp,
                             n=int(sum(map(len, groups))), stat=h, p=p))
    tab = pd.DataFrame(rows)
    tab["q"] = bh_q(tab["p"].values)
    cols = ["test", "dataset", "scope", "region", "comparison",
            "n", "stat", "p", "q"]
    return tab[cols]


# ------------------------------------------------------------- plotting -----
def fig_shannon_gradient(alphas):
    'fig_13a/b\uff1aShannon \u6cbf COMP_ORDER\uff0c\u4e09\u5730\u533a\u4e2d\u4f4d\u6570\u6298\u7ebf + \u6563\u70b9\u3002'
    for ds, letter, fn in (("16S", "a", "fig_13a_shannon_bacteria.png"),
                           ("ITS", "b", "fig_13b_shannon_fungi.png")):
        df = alphas[ds]
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        xpos = np.arange(len(COMP_ORDER))
        for reg in REGIONS:
            sub = df[df["region"] == reg]
            med = [sub.loc[sub["compartment"] == c, "Shannon"].median()
                   for c in COMP_ORDER]
            # Analysis note: see README and original laboratory scripts for methodological details.
            means, sems, dfs = [], [], []
            for c in COMP_ORDER:
                v = sub.loc[sub["compartment"] == c, "Shannon"].values
                v = v.astype(float)
                means.append(v.mean())
                sems.append(v.std(ddof=1) / np.sqrt(len(v))
                            if len(v) > 1 else 0.0)
                dfs.append(len(v) - 1)
            half = [tdist.ppf(0.975, d) * s if d > 0 else 0.0
                    for d, s in zip(dfs, sems)]
            ax.fill_between(xpos, np.asarray(means) - np.asarray(half),
                            np.asarray(means) + np.asarray(half),
                            color=C_CI, alpha=0.3, lw=0, zorder=1)
            # Analysis note: see README and original laboratory scripts for methodological details.
            for i, c in enumerate(COMP_ORDER):
                v = sub.loc[sub["compartment"] == c, "Shannon"].values
                jx = xpos[i] + RNG.uniform(-0.09, 0.09, size=len(v))
                ax.scatter(jx, v, s=7, color=COLORS[reg], alpha=0.6,
                           edgecolors="none", zorder=2)
            ax.plot(xpos, med, "-o", color=COLORS[reg], ms=4, lw=1.6,
                    label=reg, zorder=3)
        ax.set_xlabel("Compartment")
        ax.set_ylabel("Shannon index")
        ax.set_title(DATASETS[ds], pad=24)
        ax.set_xlim(-0.4, len(COMP_ORDER) - 0.6)
        set_xticks_labels(ax, list(xpos), [COMP_LABEL[c] for c in COMP_ORDER],
                          rotation=30, ha="right")
        ax.legend(frameon=False, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), columnspacing=1.0,
                  handletextpad=0.4, borderaxespad=0.0)
        panel(ax, letter)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, fn), bbox_inches="tight")
        plt.close(fig)


def fig_region_compare(alphas):
    'fig_13c\uff1abelowground/aboveground \u4e24\u7ec4 \xd7 \u4e09\u5730\u533a\u7bb1\u7ebf\u56fe\uff0c\u6807\u6ce8 KW P\u3002'
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    groups = [("Belowground", BELOW), ("Aboveground", ABOVE)]
    for ax, ds in zip(axes, ("16S", "ITS")):
        df = alphas[ds]
        pos = 0
        ticks, ticklabs = [], []
        for gname, comps in groups:
            sub = df[df["compartment"].isin(comps)]
            gname_y = -0.22
            kw_groups = [sub.loc[sub["region"] == r, "Shannon"].values
                         for r in REGIONS]
            kw_groups = [g for g in kw_groups if len(g) >= 2]
            h, pk = kruskal(*kw_groups)
            boxdata, boxcols = [], []
            for r in REGIONS:
                v = sub.loc[sub["region"] == r, "Shannon"].values
                boxdata.append(v)
                boxcols.append(COLORS[r])
                # Analysis note: see README and original laboratory scripts for methodological details.
                jx = pos + RNG.uniform(-0.1, 0.1, size=len(v))
                ax.scatter(jx, v, s=7, color=COLORS[r], alpha=0.6,
                           edgecolors="none", zorder=3)
                ticks.append(pos)
                ticklabs.append(r)
                pos += 1
            bp = ax.boxplot(boxdata, positions=np.arange(pos - 3, pos),
                            widths=0.55, patch_artist=True, showfliers=False,
                            zorder=2)
            for patch, c in zip(bp["boxes"], boxcols):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)
            for ln in bp["medians"]:
                ln.set_color("k")
                ln.set_linewidth(1.0)
            for el in ("whiskers", "caps"):
                for ln in bp[el]:
                    ln.set_color("k")
                    ln.set_linewidth(0.8)
            ax.grid(True, color="#E0E0E0", linewidth=0.5)
            ax.text(pos - 2, ax.get_ylim()[1] * 0.99,
                    "KW %s (P=%s)" % (sig_label(pk),
                                      "%.1e" % pk if pk < 0.001
                                      else "%.3f" % pk),
                    ha="center", va="top", fontsize=7)
            ax.text(pos - 2, gname_y, gname, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
            pos += 0.8
        ax.set_title(DATASETS[ds])
        ax.set_ylabel("Shannon index")
        set_xticks_labels(ax, ticks, ticklabs)
        ax.set_xlim(-0.6, pos - 0.9)
    panel(axes[0], "c")
    handles = [Patch(color=COLORS[r], label=r) for r in REGIONS]
    axes[1].legend(handles=handles, frameon=False, ncol=1,
                   loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_13c_alpha_region_compare.png"),
                bbox_inches="tight")
    plt.close(fig)


def beta_by_region(genus):
    '\u76f8\u90bb\u5bf9 \xd7 \u5730\u533a\u7684\u6837\u672c\u95f4 BC\uff08\u5c5e\u6c34\u5e73\u7edd\u5bf9\u62f7\u8d1d\u8868\uff09\u3002\u8fd4\u56de\u957f\u8868\u4e0e\u8ddd\u79bb\u65b9\u9635\u3002'
    rows, dists = [], {}
    for ds, (df, ann) in genus.items():
        dm = bc_distance(df)
        dists[ds] = (dm, ann)
        smp = list(df.index)
        idx = {s: i for i, s in enumerate(smp)}
        for reg in REGIONS:
            for a, b in ADJ_PAIRS:
                sa = [s for s in smp
                      if ann.at[s, "region"] == reg and
                      ann.at[s, "compartment"] == a]
                sb = [s for s in smp
                      if ann.at[s, "region"] == reg and
                      ann.at[s, "compartment"] == b]
                vals = [dm[idx[s1], idx[s2]] for s1 in sa for s2 in sb]
                rows.append(dict(dataset=ds, region=reg,
                                 pair="%s-%s" % (a, b),
                                 n=len(vals),
                                 mean_bc=float(np.mean(vals)) if vals else np.nan,
                                 median_bc=float(np.median(vals)) if vals else np.nan))
                rows[-1]["_vals"] = vals
    tab = pd.DataFrame(rows)
    return tab, dists


def fig_bc(tab):
    'fig_13d\uff1ax=\u76f8\u90bb\u5bf9\uff0c\u4e09\u5730\u533a\u5206\u7ec4\u7bb1\u7ebf\uff0c\u7ec6\u83cc/\u771f\u83cc\u4e24\u9762\u677f\u3002'
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    pair_keys = ["%s-%s" % (a, b) for a, b in ADJ_PAIRS]      # Analysis note: see README and original laboratory scripts for methodological details.
    pair_labs = ["%s-%s" % (COMP_LABEL[a], COMP_LABEL[b])
                 for a, b in ADJ_PAIRS]                        # Analysis note: see README and original laboratory scripts for methodological details.
    for ax, ds in zip(axes, ("16S", "ITS")):
        sub = tab[tab["dataset"] == ds]
        boxdata, boxcols, positions = [], [], []
        for pi, plab in enumerate(pair_keys):
            for ri, reg in enumerate(REGIONS):
                r = sub[(sub["pair"] == plab) & (sub["region"] == reg)]
                vals = r["_vals"].iloc[0] if len(r) else []
                boxdata.append(vals if len(vals) else [np.nan])
                boxcols.append(COLORS[reg])
                positions.append(pi + (ri - 1) * 0.26)
        bp = ax.boxplot(boxdata, positions=positions, widths=0.22,
                        patch_artist=True, showfliers=False, zorder=2)
        for patch, c in zip(bp["boxes"], boxcols):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        for ln in bp["medians"]:
            ln.set_color("k")
            ln.set_linewidth(1.0)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set_color("k")
                ln.set_linewidth(0.8)
        ax.grid(True, color="#E0E0E0", linewidth=0.5)
        ax.set_title(DATASETS[ds])
        ax.set_ylabel("Bray-Curtis distance (genus, absolute)")
        set_xticks_labels(ax, list(range(len(pair_labs))), pair_labs,
                          rotation=38, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(pair_labs) - 0.4)
    panel(axes[0], "d")
    handles = [Patch(color=COLORS[r], label=r) for r in REGIONS]
    axes[1].legend(handles=handles, frameon=False, ncol=1,
                   loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_13d_bc_by_region.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_permanova():
    'fig_13e\uff1aPERMANOVA R\xb2\uff08absolute \u53e3\u5f84\uff09\u6761\u5f62\u56fe\uff0c\u4e0d\u91cd\u7b97\u7edf\u8ba1\u3002'
    perm = pd.read_csv(PERM_PATH, sep="\t")
    perm = perm[perm["term"] != "Residual"]
    perm = perm[perm["matrix"] == "absolute"].copy()
    terms = ["compartment", "region", "compartment:region"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7), sharey=True)
    for ax, ds in zip(axes, ("16S", "ITS")):
        sub = perm[perm["dataset"] == ds]
        r2 = [float(sub.loc[sub["term"] == t, "R2"].iloc[0]) for t in terms]
        pv = [float(sub.loc[sub["term"] == t, "p"].iloc[0]) for t in terms]
        bars = ax.bar([TERM_LABEL[t] for t in terms], r2, width=0.6,
                      color=[TERM_COLOR[t] for t in terms],
                      edgecolor="k", linewidth=0.6)
        for b, v, p in zip(bars, r2, pv):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    "R²=%.2f\n%s" % (v, sig_label(p)),
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(DATASETS[ds])
        ax.set_ylim(0, max(r2) * 1.35)
        ax.set_ylabel("PERMANOVA R²")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_ha("right")
    panel(axes[0], "e")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_13e_permanova_r2.png"),
                bbox_inches="tight")
    plt.close(fig)


def write_hellinger(genus):
    '\u6837\u672c\xd7\u5c5e Hellinger\uff1asqrt(abs/\u884c\u548c)\uff0c\u884c\u548c\u4e3a 0 \u7f6e 0\u3002'
    for ds, (df, _ann) in genus.items():
        rs = df.sum(axis=1)
        rel = df.div(rs.replace(0, np.nan), axis=0).fillna(0.0)
        hel = np.sqrt(rel)
        out = os.path.join(R_IN, "hellinger_genus_%s.tsv" % ds)
        hel.round(6).to_csv(out, sep="\t", index_label="sample_id")


# ------------------------------------------------------------------ main ----
def main():
    alphas = {ds: load_alpha(ds) for ds in DATASETS}
    genus = {ds: load_genus_abs(ds) for ds in DATASETS}

    # 1) fig a/b
    fig_shannon_gradient(alphas)

    # 2) tab 13
    tab = alpha_tests(alphas)
    tab.to_csv(os.path.join(TABLES, "tab_13_alpha_tests.tsv"),
               sep="\t", index=False)

    # 3) fig c
    fig_region_compare(alphas)

    # Analysis note: see README and original laboratory scripts for methodological details.
    bc_tab, _dists = beta_by_region(genus)
    bc_tab.drop(columns=["_vals"]).to_csv(
        os.path.join(TABLES, "tab_13_bc_by_region.tsv"),
        sep="\t", index=False)
    fig_bc(bc_tab)

    # 5) fig e
    fig_permanova()

    # 6) hellinger
    write_hellinger(genus)

    # ------------------------------------------------------------ summary ----
    print('=== tab_13_alpha_tests: \u663e\u8457 (q<0.05) \u6761\u76ee ===')
    sig = tab[tab["q"] < 0.05]
    print(sig.groupby(["test", "dataset", "scope"]).size())
    kw = tab[tab["test"] == "KW"]
    print('\nKW\uff08\u5730\u533a\u95f4\uff0c\u5404 compartment\uff09\uff1a')
    print(kw[["dataset", "comparison", "p", "q"]].to_string(index=False))
    print('\n=== BC\uff1a\u5404\u5730\u533a\u5e73\u5747\u6700\u5927\u7684\u76f8\u90bb\u5bf9 ===')
    for ds in DATASETS:
        sub = bc_tab[bc_tab["dataset"] == ds]
        agg = sub.groupby("pair")["mean_bc"].mean().sort_values(ascending=False)
        print(ds, dict(agg.round(3)), sep=": ")
    print("\nDone.")


if __name__ == "__main__":
    main()
