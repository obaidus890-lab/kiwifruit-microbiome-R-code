# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")

from analysis_configuration import FIGS, TABLES
import analysis_helpers
from analyze_fruit_metabolome_microbiome_associations import (load_metabolome, sample_map, microbe_matrix,
                                phylum_loads, shannon, total_load, mantel,
                                partial_mantel, bc_dist, euclid_dist,
                                rda_adjusted_r2, META_DIR)

SITE_COL = {"WH": "#E64B35", "BZ": "#3C5488", "SX": "#00A087"}
BAC_RED, FUN_BLUE = "#E64B35", "#3C5488"


def hellinger(X):
    pr = X.div(X.sum(axis=1), axis=0)
    return np.sqrt(pr.fillna(0))


def pca_full(C):
    '\u4e2d\u5fc3\u5316\u77e9\u9635 -> (site scores, eig)\uff08\u4fdd\u7559\u5168\u90e8\u8f74\uff09\u3002'
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    return u * s[None, :], s ** 2


def proc_full(A, B, nperm=999, seed=123):
    'vegan procrustes(symmetric) + protest \u7b49\u4ef7\uff1b\u8fd4\u56de r, M2, p, R, Yrot\u3002'
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    u, s, vt = np.linalg.svd(A.T.dot(B))
    R = u.dot(vt)
    Yrot = A.dot(R)
    m2 = float(((B - Yrot) ** 2).sum())
    r = np.sqrt(max(0.0, 1 - m2))
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(nperm):
        Ap = A[rng.permutation(len(A))]
        u2, s2, vt2 = np.linalg.svd(Ap.T.dot(B))
        m2p = float(((B - Ap.dot(u2.dot(vt2))) ** 2).sum())
        if m2p <= m2:
            cnt += 1
    return r, m2, (cnt + 1) / (nperm + 1), R, Yrot


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    met, groups, fmap = sample_map()
    X = met.T.loc[sorted(fmap.keys())]          # Analysis note: see README and original laboratory scripts for methodological details.
    g = pd.Series({s: groups[s] for s in X.index})
    pm = X[X > 0].values.min() / 2
    L = np.log2(X.where(X > 0, pm))             # Analysis note: see README and original laboratory scripts for methodological details.
    U = ((L - L.mean()) / L.std(ddof=1)).fillna(0)   # UV autoscaling
    C = U.values

    inv = {v: k for k, v in fmap.items()}
    samples = [fmap[s] for s in X.index]
    d16 = analysis_helpers.load_dataset("16S")
    dit = analysis_helpers.load_dataset("ITS")
    Xb = microbe_matrix(d16, samples, qmp=True)
    Xf = microbe_matrix(dit, samples, qmp=True)

    # Analysis note: see README and original laboratory scripts for methodological details.
    rows_proc = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, (king, Xm) in zip(axes, [("bacteria", Xb), ("fungi", Xf)]):
        H = hellinger(Xm)
        Hc = (H - H.mean()).values
        otu_sc, _ = pca_full(Hc)
        met_sc, _ = pca_full(C)
        r, m2, p, R, Yrot = proc_full(otu_sc, met_sc)
        An = otu_sc / np.linalg.norm(otu_sc)
        Bn = met_sc / np.linalg.norm(met_sc)
        Yrot = An.dot(R)              # Analysis note: see README and original laboratory scripts for methodological details.
        rows_proc.append(dict(scope="pooled", kingdom=king, view="QMP",
                              procrustes_r=r, M2=m2, p=p))
        for s in ["WH", "BZ", "SX"]:
            msk = g == s
            ax.scatter(Yrot[msk, 0], Yrot[msk, 1], s=26, c=SITE_COL[s],
                       marker="o", zorder=3)
            ax.scatter(Bn[msk, 0], Bn[msk, 1], s=26, c=SITE_COL[s],
                       marker="^", zorder=3)
            for i in np.where(msk)[0]:
                ax.annotate("", (Bn[i, 0], Bn[i, 1]),
                            (Yrot[i, 0], Yrot[i, 1]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.5,
                                            color=SITE_COL[s]), zorder=2)
        ax.axhline(0, ls="--", lw=0.4, c="grey")
        ax.axvline(0, ls="--", lw=0.4, c="grey")
        ax.set_title("%s: Procrustes r=%.3f, P=%.3f" % (king, r, p),
                     fontsize=8)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    axes[0].text(-0.12, 1.05, "a", transform=axes[0].transAxes, fontsize=14,
                 fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_a.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for scope in ["WH", "BZ", "SX"]:
        idx = [s for s in X.index if groups[s] == scope]
        Us = ((L.loc[idx] - L.loc[idx].mean()) /
              L.loc[idx].std(ddof=1)).fillna(0).values
        dsamp = [fmap[s] for s in idx]
        for king, dat in [("bacteria", d16), ("fungi", dit)]:
            Xm = microbe_matrix(dat, dsamp, qmp=True)
            Hc = (hellinger(Xm) - hellinger(Xm).mean()).values
            os_, _ = pca_full(Hc)
            ms_, _ = pca_full(Us)
            k = min(8, os_.shape[1] - 1, ms_.shape[1] - 1)
            r, m2, p, _, _ = proc_full(os_[:, :k], ms_[:, :k], nperm=499)
            rows_proc.append(dict(scope=scope, kingdom=king, view="QMP",
                                  procrustes_r=r, M2=m2, p=p))
    pd.DataFrame(rows_proc).to_csv(
        os.path.join(TABLES, "tab_sec16_qmp_procrustes.tsv"), sep="\t",
        index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    dm_met = euclid_dist(U)
    dm_b = bc_dist(Xb)
    dm_f = bc_dist(Xf)
    r_b, p_b = mantel(dm_met, dm_b, nperm=999)
    r_f, p_f = mantel(dm_met, dm_f, nperm=999)
    pr_b, pp_b = partial_mantel(dm_met, dm_b, dm_f, nperm=999)
    pr_f, pp_f = partial_mantel(dm_met, dm_f, dm_b, nperm=999)
    rows_mant = [dict(scope="pooled", kingdom="bacteria", r=r_b, p=p_b,
                      partial_r=pr_b, partial_p=pp_b),
                 dict(scope="pooled", kingdom="fungi", r=r_f, p=p_f,
                      partial_r=pr_f, partial_p=pp_f)]
    sx_idx = [s for s in X.index if g[s] == "SX"]
    Usx = ((L.loc[sx_idx] - L.loc[sx_idx].mean()) /
           L.loc[sx_idx].std(ddof=1)).fillna(0)
    dm_sx = euclid_dist(Usx)
    dm_bsx = bc_dist(Xb.loc[[fmap[s] for s in sx_idx]])
    dm_fsx = bc_dist(Xf.loc[[fmap[s] for s in sx_idx]])
    r_bs, p_bs = mantel(dm_sx, dm_bsx, nperm=999)
    r_fs, p_fs = mantel(dm_sx, dm_fsx, nperm=999)
    pr_bs, pp_bs = partial_mantel(dm_sx, dm_bsx, dm_fsx, nperm=999)
    pr_fs, pp_fs = partial_mantel(dm_sx, dm_fsx, dm_bsx, nperm=999)
    rows_mant += [dict(scope="SX", kingdom="bacteria", r=r_bs, p=p_bs,
                       partial_r=pr_bs, partial_p=pp_bs),
                  dict(scope="SX", kingdom="fungi", r=r_fs, p=p_fs,
                       partial_r=pr_fs, partial_p=pp_fs)]
    pd.DataFrame(rows_mant).to_csv(
        os.path.join(TABLES, "tab_sec16_qmp_mantel.tsv"), sep="\t",
        index=False)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    panels = [
        (dm_b, dm_met, r_b, p_b, pr_b, BAC_RED, "pooled bacteria"),
        (dm_f, dm_met, r_f, p_f, pr_f, FUN_BLUE, "pooled fungi"),
        (dm_bsx, dm_sx, r_bs, p_bs, pr_bs, BAC_RED, "SX bacteria"),
        (dm_fsx, dm_sx, r_fs, p_fs, pr_fs, FUN_BLUE, "SX fungi")]
    for ax, (dmx, dmy, r, p, pr_, col, ttl) in zip(axes, panels):
        y = np.asarray(dmy)
        x = np.asarray(dmx)
        ax.scatter(x, y, s=6, alpha=0.25, c="grey")
        z = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, np.polyval(z, xx), color=col, lw=1.4)
        ax.set_xlabel("Bray-Curtis distance", fontsize=6)
        ax.set_ylabel("Euclidean distance\nof metabolites", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_title("%s\nMantel r=%.3f, P=%.3f; partial r=%.3f" % (
            ttl, r, p, pr_), fontsize=6.5)
    axes[0].text(-0.25, 1.08, "b", transform=axes[0].transAxes, fontsize=14,
                 fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_b.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    fac = pd.DataFrame(index=X.index)
    fac["16S_copies"] = np.log10(total_load(d16, samples) + 1)
    fac["ITS_copies"] = np.log10(total_load(dit, samples) + 1)
    fac["Shannon_bacteria"] = np.asarray(shannon(Xb))
    fac["Shannon_fungi"] = np.asarray(shannon(Xf))
    pb = phylum_loads(d16, samples, qmp=True, top=5)
    pf = phylum_loads(dit, samples, qmp=True, top=5)
    for c in pb.columns:
        fac["B_" + str(c)] = np.log10(pb[c].values + 1)
    for c in pf.columns:
        fac["F_" + str(c)] = np.log10(pf[c].values + 1)
    KINGMAP = {"16S_copies": "bacteria", "ITS_copies": "fungi",
               "Shannon_bacteria": "bacteria", "Shannon_fungi": "fungi"}
    is_bac = {c: KINGMAP.get(c, str(c).startswith("B_"))
              for c in fac.columns}
    rows_rda = []
    sel = []
    for c in fac.columns:
        r2, adj, p = rda_adjusted_r2(C, fac[[c]].values, nperm=199)
        rows_rda.append(dict(factor=c,
                             kingdom="bacteria" if is_bac[c] else "fungi",
                             marginal_r2=r2, marginal_adj=adj, marginal_p=p,
                             selected=False))
        if p < 0.10:
            sel.append(c)
    r2f, adjf, pf = rda_adjusted_r2(C, fac[sel].values, nperm=199)
    for rr in rows_rda:
        rr["selected"] = rr["factor"] in sel
        rr["kingdom"] = "bacteria" if is_bac[rr["factor"]] else "fungi"
    pd.DataFrame(rows_rda).to_csv(
        os.path.join(TABLES, "tab_sec16_qmp_rda.tsv"), sep="\t",
        index=False)
    # Analysis note: see README and original laboratory scripts for methodological details.
    from sklearn.linear_model import LinearRegression
    Y = C
    Xs = fac[sel].values
    reg = LinearRegression().fit(Xs, Y)
    Yhat = reg.predict(Xs)
    u, s, vt = np.linalg.svd(Yhat, full_matrices=False)
    site = u[:, :2] * s[:2]
    eig_pct = s ** 2 / (s ** 2).sum() * 100
    B = np.linalg.lstsq(site, Xs, rcond=None)[0].T          # Analysis note: see README and original laboratory scripts for methodological details.
    B = B / np.abs(B).max() * np.abs(site).max() * 0.7
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    # Analysis note: see README and original laboratory scripts for methodological details.
    for sp in ["WH", "BZ", "SX"]:
        msk = (g == sp).values
        xs, ys = site[msk, 0], site[msk, 1]
        ax.scatter(xs, ys, s=26, c=SITE_COL[sp], alpha=0.85,
                   edgecolor="white", linewidth=0.3, label=sp, zorder=4)
        if len(xs) >= 3:
            ax.add_patch(Ellipse((xs.mean(), ys.mean()),
                                 2 * xs.std(ddof=1), 2 * ys.std(ddof=1),
                                 edgecolor=SITE_COL[sp],
                                 facecolor=SITE_COL[sp], alpha=0.13,
                                 lw=0.8, zorder=1))
    # Analysis note: see README and original laboratory scripts for methodological details.
    for j, c in enumerate(sel):
        col = BAC_RED if is_bac[c] else FUN_BLUE
        ax.annotate("", (B[j, 0], B[j, 1]), (0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color=col,
                                    mutation_scale=11))
        ax.annotate(c, xy=(B[j, 0], B[j, 1]),
                    xytext=(10, (j - (len(sel) - 1) / 2.0) * 9),
                    textcoords="offset points", fontsize=6, color=col,
                    ha="left", va="center")
    ax.axhline(0, ls="--", lw=0.6, c="k")
    ax.axvline(0, ls="--", lw=0.6, c="k")
    ax.set_xlabel("RDA1 (%.2f%%)" % eig_pct[0], fontsize=7)
    ax.set_ylabel("RDA2 (%.2f%%)" % eig_pct[1], fontsize=7)
    ax.set_title("RDA: R2=%.3f, adj=%.3f, P=%.3f" % (r2f, adjf, pf),
                 fontsize=8)
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=SITE_COL[sp], markersize=6, label=sp)
               for sp in ["WH", "BZ", "SX"]]
    handles += [Line2D([0], [0], color=BAC_RED, lw=1.4,
                       label="bacterial factor"),
                Line2D([0], [0], color=FUN_BLUE, lw=1.4,
                       label="fungal factor")]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="best")
    ax.text(-0.1, 1.05, "c", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_c.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    from sklearn.ensemble import RandomForestRegressor
    subclass = (pd.read_csv(os.path.join(META_DIR, "chemical_class.csv"),
                            encoding="gbk")
                .set_index("Metabolite")["Subclass"].reindex(L.columns)
                .fillna("Unclassified"))
    CL = U.groupby(subclass, axis=1).mean()      # Analysis note: see README and original laboratory scripts for methodological details.
    print("metabolite classes:", CL.shape[1])

    def oob_pct_inc(rf, Xa, y, seed=1):
        'R randomForest importance(type=1) \u7b49\u4ef7\uff1aOOB %IncMSE\u3002'
        from sklearn.ensemble._forest import _generate_unsampled_indices
        rng = np.random.RandomState(seed)
        inc = np.zeros(Xa.shape[1])
        cnt = 0
        for tree in rf.estimators_:
            oob = _generate_unsampled_indices(tree.random_state, len(y),
                                              len(y))
            if len(oob) < 3:
                continue
            Xo, yo = Xa[oob], y[oob]
            val = tree.tree_.value[:, 0, 0]
            base = ((val[tree.apply(Xo)] - yo) ** 2).mean()
            if base <= 1e-12:
                continue
            for j in range(Xa.shape[1]):
                Xp = Xo.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xo)), j]
                inc[j] += (((val[tree.apply(Xp)] - yo) ** 2).mean() -
                           base) / base * 100.0
            cnt += 1
        return inc / max(cnt, 1)

    tops = {}
    for king, Xm in [("bacteria", Xb), ("fungi", Xf)]:
        Hc = (hellinger(Xm) - hellinger(Xm).mean()).values
        pc1 = pca_full(Hc)[0][:, 0]
        rf = RandomForestRegressor(n_estimators=500, random_state=1)
        rf.fit(CL.values, pc1)
        imp = pd.Series(oob_pct_inc(rf, CL.values, pc1), index=CL.columns)
        tops[king] = imp.sort_values(ascending=False)
    tb, tf = tops["bacteria"].head(20), tops["fungi"].head(20)
    inter = sorted(set(tb.index) & set(tf.index))
    rows_rf = []
    for king, t in [("bacteria", tb), ("fungi", tf)]:
        for i, (mname, v) in enumerate(t.items()):
            rows_rf.append(dict(kingdom=king, rank=i + 1,
                                Metabolite_class=mname,
                                RF_pctIncMSE=v,
                                rank_fungi=tf.index.get_loc(mname) + 1
                                if mname in tf.index else np.nan,
                                rank_bacteria=tb.index.get_loc(mname) + 1
                                if mname in tb.index else np.nan))
    pd.DataFrame(rows_rf).to_csv(
        os.path.join(TABLES, "tab_sec16_qmp_rf_top.tsv"), sep="\t",
        index=False)
    pd.DataFrame(dict(metabolite=inter)).to_csv(
        os.path.join(TABLES, "tab_sec16_qmp_rf_intersection.tsv"), sep="\t",
        index=False)

    def short(s, n=34):
        return s if len(s) <= n else s[:n] + "…"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.2),
                             gridspec_kw=dict(width_ratios=[1, 0.35, 1]))
    a0, a1, a2 = axes
    yb = np.arange(len(tb))[::-1]
    a0.barh(yb, -tb.values, height=0.7, color="#7EB4C6")
    a0.set_yticks(yb)
    a0.set_yticklabels([short(m) for m in tb.index], fontsize=5.6)
    a0.yaxis.tick_right()
    a0.set_xlim(-tb.max() * 1.15, 0)
    # Analysis note: see README and original laboratory scripts for methodological details.
    a0.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % abs(v)))
    a0.set_xlabel("%IncMSE", fontsize=7)
    a0.set_title("bacteria", fontsize=8)
    yf = np.arange(len(tf))[::-1]
    a2.barh(yf, tf.values, height=0.7, color="#AD93B8")
    a2.set_yticks(yf)
    a2.set_yticklabels([short(m) for m in tf.index], fontsize=5.6)
    a2.set_xlim(0, tf.max() * 1.15)
    a2.set_xlabel("%IncMSE", fontsize=7)
    a2.set_title("fungi", fontsize=8)
    a1.set_ylim(-0.5, max(len(tb), len(tf)) - 0.5)
    a1.set_xlim(0, 1)
    for m in inter:
        y1 = yb[list(tb.index).index(m)]
        y2 = yf[list(tf.index).index(m)]
        a1.plot([0, 1], [y1, y2], lw=0.7, color="red", zorder=1)
        a1.scatter([0], [y1], s=14, c="#7EB4C6", zorder=2)
        a1.scatter([1], [y2], s=14, c="#AD93B8", zorder=2)
    a1.set_xticks([])
    a1.set_yticks([])
    if not inter:
        a1.text(0.5, 0.5, "no shared\ntop-20 metabolites", ha="center",
                va="center", fontsize=6.5, color="grey")
    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(False)
    a0.text(-0.15, 1.03, "d", transform=a0.transAxes, fontsize=14,
            fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sec16_d.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    from infer_fruit_bacteria_fungi_cooccurrence_network import combined_qmp, build_network, \
        module_eigengene_net
    from networkx.algorithms import community as nxcom
    Z = combined_qmp(fmap)
    G = build_network(Z)
    comps = sorted(nxcom.greedy_modularity_communities(G), key=len,
                   reverse=True)
    meigs = {}
    for mi, c in enumerate(comps):
        c = list(c)
        if len(c) < 3:
            continue
        meigs["M%d" % (mi + 1)] = module_eigengene_net(
            Z.loc[sorted(Z.index)], c)
    mods = list(meigs)
    sub2cls = (pd.read_csv(os.path.join(META_DIR, "chemical_class.csv"),
                           encoding="gbk").drop_duplicates("Subclass")
               .set_index("Subclass")["Class"])
    pairs = []
    for m in mods:
        for metb in CL.columns:
            r, p = spearmanr(meigs[m], CL[metb])
            pairs.append(dict(net_module=m, metabolite_class=metb,
                              superclass=sub2cls.get(metb, "Others"),
                              rho=r, p=p))
    mdf = pd.DataFrame(pairs)
    _, fdr, _, _ = multipletests(mdf.p, method="fdr_bh")
    mdf["FDR"] = fdr
    sig = (mdf.p < 0.05) & (mdf.rho.abs() >= 0.5)
    keep = sorted(set(mdf.loc[sig, "metabolite_class"]))
    sub = mdf[mdf.metabolite_class.isin(keep)]
    sub.to_csv(os.path.join(TABLES, "tab_sec16_netmodule_metabolite.tsv"),
               sep="\t", index=False)
    # Analysis note: see README and original laboratory scripts for methodological details.
    topk = (sub.sort_values("rho", key=lambda s: s.abs(), ascending=False)
            .groupby("net_module").head(15))
    keep = sorted(set(topk.metabolite_class))
    sub = mdf[mdf.metabolite_class.isin(keep)]

    if len(keep):
        piv_r = sub.pivot_table(index="metabolite_class",
                                columns="net_module", values="rho")
        piv_r = piv_r.reindex(columns=mods)
        piv_p = sub.pivot_table(index="metabolite_class",
                                columns="net_module", values="p")
        piv_p = piv_p.reindex(columns=mods)
        clv = (sub.drop_duplicates("metabolite_class")
               .set_index("metabolite_class")["superclass"])
        order = []
        for c in clv.value_counts().index:
            order += [m for m in clv[clv == c].index
                      if m in piv_r.index]
        piv_r, piv_p = piv_r.loc[order], piv_p.loc[order]
        clv = clv.loc[order]
        PAL = ["#B9717D", "#F5C8B5", "#AD93B8", "#7EB4C6", "#98CA8B",
               "#795B34", "#E06E6E", "#6E86AC", "#DAA520", "#00A087",
               "#3C5488", "#999999"]
        ccols = {c: PAL[i % len(PAL)] for i, c in
                 enumerate(clv.value_counts().index)}
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "qmp", ["#240EFF", "#B491F9", "#ECEAEF", "#FFA48C", "#FF1A09"])
        fig, ax = plt.subplots(figsize=(3.4, 0.32 * len(order) + 1.8))
        for i, m in enumerate(order):
            for j, mm in enumerate(mods):
                r, p = piv_r.values[i, j], piv_p.values[i, j]
                if p < 0.05:
                    col = cmap((r + 1) / 2)
                else:
                    col = "#D9D9D9"
                ax.add_patch(plt.Rectangle((j, len(order) - 1 - i), 1, 1,
                                           facecolor=col, edgecolor="white",
                                           lw=0.8))
            ax.add_patch(plt.Rectangle((-0.6, len(order) - 1 - i), 0.5, 1,
                                       facecolor=ccols[clv[m]],
                                       edgecolor="white", lw=0.5))
        ax.set_xlim(-0.7, len(mods))
        ax.set_ylim(0, len(order))
        ax.set_xticks([j + 0.5 for j in range(len(mods))])
        ax.set_xticklabels(mods, fontsize=7)
        ax.set_yticks([i + 0.5 for i in range(len(order))])
        ax.set_yticklabels([short(m, 30) for m in order], fontsize=5.4)
        ax.xaxis.set_ticks_position("top")
        ax.set_title("Network modules x metabolite classes "
                     "(grey = P>=0.05)", fontsize=8)
        ax.text(-0.35, 1.02, "i", transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig_sec16_i.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    # Analysis note: see README and original laboratory scripts for methodological details.
    print("== Procrustes (PCA-hellinger, full axes) ==")
    pr = pd.DataFrame(rows_proc)
    print(pr[pr.scope == "pooled"].round(3).to_string(index=False))
    print("== Mantel (QMP, UV metabolome) ==")
    print(pd.DataFrame(rows_mant).round(3).to_string(index=False))
    print("== RDA selected ==", sel)
    print("full R2=%.4f adj=%.4f p=%.3f; axes %.2f/%.2f" %
          (r2f, adjf, pf, eig_pct[0], eig_pct[1]))
    print(pd.DataFrame(rows_rda).round(3).to_string(index=False))
    print("== RF top5 ==")
    print("bacteria:", list(tb.head(5).index))
    print("fungi:", list(tf.head(5).index))
    print("intersection(%d):" % len(inter), inter)
    print("== Fig6B rows ==", len(keep))
    if len(keep):
        sg = sub[(sub.p < 0.05)]
        print("sig cells:", len(sg), " pos:", int((sg.rho > 0).sum()),
              " neg:", int((sg.rho < 0).sum()))
        print(sg.sort_values("p").head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
