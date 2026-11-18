# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

from analysis_configuration import FIGS, TABLES
import analysis_helpers
from analyze_fruit_metabolome_microbiome_associations import sample_map, microbe_matrix
from analyze_fruit_metabolome_modules_and_source_tracking import module_eigengenes

CORE_PREV = 0.7


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    met, groups, fmap = sample_map()
    X = met.T.loc[sorted(fmap.keys())]
    pm = X[X > 0].values.min() / 2
    L = np.log2(X.where(X > 0, pm))
    U = ((L - L.mean()) / L.std(ddof=1)).fillna(0)
    pc1 = np.linalg.svd(U.values, full_matrices=False)[0][:, 0]

    eig = module_eigengenes()[0]
    mepink = eig["MEpink"].reindex(X.index).values

    samples = [fmap[s] for s in X.index]
    d16 = analysis_helpers.load_dataset("16S")
    dit = analysis_helpers.load_dataset("ITS")

    nodes = pd.read_csv(os.path.join(TABLES, "tab_sec16_crossking_nodes.tsv"),
                        sep="\t")
    nmap = {(r.kingdom, r.genus): r for r in nodes.itertuples()}

    # Analysis note: see README and original laboratory scripts for methodological details.
    strat = {}
    for king, fn in [("bacteria", "tab_sec15_strategy_16S.tsv"),
                     ("fungi", "tab_sec15_strategy_ITS.tsv")]:
        st = pd.read_csv(os.path.join(TABLES, fn), sep="\t")
        strat[king] = st.drop_duplicates("genus").set_index("genus")["strategy"]

    rows_taxa, sets = [], {}
    for king, dat in [("bacteria", d16), ("fungi", dit)]:
        Xm = microbe_matrix(dat, samples, qmp=True)
        Xm = Xm.loc[[fmap[s] for s in X.index]]
        prev = (Xm > 0).mean()
        ab = Xm.mean()
        Z = np.log10(Xm + 1)
        core = prev[prev >= CORE_PREV].index.tolist()
        noncore = [c for c in Xm.columns if c not in core]
        sets[(king, "core")] = Z[core]
        sets[(king, "noncore")] = Z[noncore]
        sets[(king, "all")] = Z
        for gname in Xm.columns:
            nd = nmap.get((king, gname))
            rows_taxa.append(dict(
                kingdom=king, genus=gname, prevalence=round(prev[gname], 3),
                mean_qmp=round(ab[gname], 1),
                core=gname in core,
                net_module=nd.module if nd is not None else "",
                key_type=nd.key if nd is not None else "",
                strategy=strat[king].get(gname, "")))
    taxa = pd.DataFrame(rows_taxa)
    taxa.to_csv(os.path.join(TABLES, "tab_sec16_core_taxa.tsv"), sep="\t",
                index=False)
    nb = int((taxa.kingdom == "bacteria").sum())
    cb = int(taxa[(taxa.kingdom == "bacteria") & taxa.core].shape[0])
    cf = int(taxa[(taxa.kingdom == "fungi") & taxa.core].shape[0])
    print("core taxa: bacteria %d/%d, fungi %d/%d" % (
        cb, nb, cf, len(taxa) - nb))

    # Analysis note: see README and original laboratory scripts for methodological details.
    for m in ["M1", "M2", "M3"]:
        sub = taxa[taxa.net_module == m]
        if len(sub) == 0:
            continue
        print("module %s: n=%d, core%%=%.1f" % (
            m, len(sub), 100.0 * sub.core.mean()))
    key = taxa[taxa.key_type.notna() & (taxa.key_type != "")]
    print("key taxa: n=%d, core%%=%.1f" % (len(key), 100.0 * key.core.mean()))

    # Analysis note: see README and original laboratory scripts for methodological details.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    def loo_r2(P, y):
        if P.shape[1] == 0:
            return np.nan
        rf = RandomForestRegressor(n_estimators=500, random_state=1)
        pred = cross_val_predict(rf, P.values, y, cv=LeaveOneOut())
        ss = ((y - y.mean()) ** 2).sum()
        return 1 - ((y - pred) ** 2).sum() / ss

    rows_rf = []
    for king in ["bacteria", "fungi"]:
        for setname in ["core", "noncore", "all"]:
            P = sets[(king, setname)]
            for resp, y in [("metPC1", pc1), ("MEpink", mepink)]:
                r2 = loo_r2(P, y)
                rows_rf.append(dict(kingdom=king, predictor_set=setname,
                                    n_genera=P.shape[1], response=resp,
                                    LOO_R2=round(r2, 3)))
                print("RF %s %-7s %-8s %-6s n=%3d LOO R2=%.3f" % (
                    king, setname, resp, "", P.shape[1], r2))
    rf = pd.DataFrame(rows_rf)
    rf.to_csv(os.path.join(TABLES, "tab_sec16_core_rf.tsv"), sep="\t",
              index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    def short(s, n=20):
        return s if len(s) <= n else s[:n] + "…"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, king in zip(axes, ["bacteria", "fungi"]):
        sub = taxa[taxa.kingdom == king]
        nc, cc = sub[~sub.core], sub[sub.core]
        ax.scatter(np.log10(nc.mean_qmp + 1), nc.prevalence, s=14,
                   c="#9aa4b2", alpha=0.7, label="non-core", zorder=2)
        ax.scatter(np.log10(cc.mean_qmp + 1), cc.prevalence, s=20,
                   c="#E64B35" if king == "bacteria" else "#3C5488",
                   alpha=0.85, label="core (prev>=%.1f)" % CORE_PREV,
                   zorder=3)
        ax.axhline(CORE_PREV, ls="--", lw=0.6, c="grey")
        top = cc.sort_values("mean_qmp", ascending=False).head(6)
        for ii, r in enumerate(top.itertuples()):
            ax.annotate(short(r.genus), (np.log10(r.mean_qmp + 1),
                                         r.prevalence), fontsize=5.6,
                        xytext=(3, 2 - ii * 7), textcoords="offset points")
        ax.set_xlabel("log10 mean absolute abundance (copies/g + 1)",
                      fontsize=7)
        ax.set_ylabel("prevalence across 27 fruit samples", fontsize=7)
        ax.set_title("%s (%d core / %d)" % (king, len(cc), len(sub)),
                     fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=6)
    axes[0].text(-0.12, 1.05, "j", transform=axes[0].transAxes, fontsize=14,
                 fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_j.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, resp in zip(axes, ["metPC1", "MEpink"]):
        sub = rf[rf.response == resp]
        xpos = np.arange(2)
        w = 0.25
        for i, setname in enumerate(["core", "noncore", "all"]):
            vals = [sub[(sub.kingdom == k) & (sub.predictor_set == setname)]
                    ["LOO_R2"].values[0] for k in ["bacteria", "fungi"]]
            ax.bar(xpos + (i - 1) * w, vals, width=w,
                   color=["#E64B35", "#3C5488", "#999999"][i],
                   label=setname)
        ax.axhline(0, lw=0.6, c="k")
        ax.set_xticks(xpos)
        ax.set_xticklabels(["bacteria", "fungi"], fontsize=7)
        ax.set_ylabel("LOO random-forest R2", fontsize=7)
        ax.set_title("response: %s" % resp, fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=6)
    axes[0].text(-0.12, 1.08, "k", transform=axes[0].transAxes, fontsize=14,
                 fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_k.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("-> fig_sec16_j/k.png, tab_sec16_core_taxa.tsv, tab_sec16_core_rf.tsv")


if __name__ == "__main__":
    main()
