# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nxcom
from networkx.algorithms.community import modularity as nx_modularity
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")

from analysis_configuration import FIGS, TABLES
from analyze_fruit_metabolome_modules_and_source_tracking import module_eigengenes
from analyze_fruit_metabolome_microbiome_associations import microbe_matrix
import analysis_helpers

R_TH, P_TH = 0.6, 0.001
NAMES = ["turquoise", "blue", "brown", "yellow", "green", "red",
         "black", "pink", "magenta", "purple"]


def combined_qmp(fmap):
    '\u5408\u5e76\u4e24\u754c\u679c\u5b9e QMP \u5c5e\u77e9\u9635\uff08\u5217\u524d\u7f00 B|/F|\uff0c\u884c=\u4ee3\u8c22\u7ec4\u6837\u672c\u540d\uff09\u3002'
    inv = {v: k for k, v in fmap.items()}
    parts = []
    for ds, pref in [("16S", "B"), ("ITS", "F")]:
        dat = analysis_helpers.load_dataset(ds)
        X = microbe_matrix(dat, list(inv.keys()), qmp=True)
        X.index = [inv[s] for s in X.index]
        X = np.log10(X + 1)
        X.columns = ["%s|%s" % (pref, c) for c in X.columns]
        parts.append(X)
    Z = pd.concat(parts, axis=1)
    return Z.loc[sorted(Z.index)]


def build_network(Z):
    nodes = list(Z.columns)
    R = np.zeros((len(nodes), len(nodes)))
    P = np.ones((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            r, p = spearmanr(Z[nodes[i]], Z[nodes[j]])
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(R[i, j]) > R_TH and P[i, j] < P_TH:
                G.add_edge(nodes[i], nodes[j],
                           rho=R[i, j], p=P[i, j],
                           sign=int(np.sign(R[i, j])))
    return G


def module_eigengene_net(Z, members):
    sub = Z[members]
    s = (sub - sub.mean()) / sub.std(ddof=1)
    s = s.fillna(0)
    u, sv, vt = np.linalg.svd(s.values, full_matrices=False)
    e = u[:, 0] * np.sign(vt[0].sum())
    return pd.Series(e, index=sub.index)


def main():
    eig, met, fmap = module_eigengenes()
    Z = combined_qmp(fmap)
    G = build_network(Z)

    comps = list(nxcom.greedy_modularity_communities(G))
    comps = sorted(comps, key=len, reverse=True)
    mod_of = {}
    for mi, c in enumerate(comps):
        for n in c:
            mod_of[n] = "M%d" % (mi + 1)
    deg = dict(G.degree())
    n_pos = sum(1 for _, _, d in G.edges(data=True) if d["sign"] > 0)
    n_neg = G.number_of_edges() - n_pos
    n_cross = sum(1 for a, b in G.edges() if a.split("|")[0] != b.split("|")[0])

    # Analysis note: see README and original laboratory scripts for methodological details.
    key = {}
    for ds, pref in [("16S", "B"), ("ITS", "F")]:
        k = pd.read_csv(os.path.join(TABLES, "tab_sec15_keystone_%s.tsv" % ds),
                        sep="\t")
        f = k[k.scope == "F"]
        for _, r in f.iterrows():
            tag = []
            if bool(r.keystone):
                tag.append("keystone")
            if str(r.hub_type).endswith("hub"):
                tag.append("hub")
            if str(r.indval_compartment) == "F":
                tag.append("indicator")
            if tag:
                key["%s|%s" % (pref, r.genus)] = "+".join(tag)

    # Analysis note: see README and original laboratory scripts for methodological details.
    topo_rows = [dict(entity="network", nodes=G.number_of_nodes(),
                      edges=G.number_of_edges(), pos=n_pos, neg=n_neg,
                      modularity=round(nx_modularity(G, comps), 3),
                      n_modules=len([c for c in comps if len(c) >= 2]),
                      cross_edges=n_cross,
                      cross_frac=round(n_cross / G.number_of_edges(), 3))]
    for mi, c in enumerate(comps):
        c = list(c)
        if len(c) < 2:
            continue
        nb = sum(1 for n in c if n.startswith("B|"))
        nf = len(c) - nb
        sg = G.subgraph(c)
        ce = sum(1 for a, b in sg.edges()
                 if a.split("|")[0] != b.split("|")[0])
        topo_rows.append(dict(entity="M%d" % (mi + 1), nodes=len(c),
                              edges=int(sg.number_of_edges()),
                              pos=sum(1 for a, b, d in sg.edges(data=True)
                                      if d["sign"] > 0),
                              neg=sum(1 for a, b, d in sg.edges(data=True)
                                      if d["sign"] < 0),
                              n_bact=nb, n_fung=nf, cross_edges=ce))
    topo = pd.DataFrame(topo_rows)
    topo.to_csv(os.path.join(TABLES, "tab_sec16_crossking_topo.tsv"),
                sep="\t", index=False)

    nodes = pd.DataFrame([
        dict(node=n, kingdom="bacteria" if n.startswith("B|") else "fungi",
             genus=n.split("|", 1)[1], module=mod_of.get(n, "singleton"),
             degree=deg[n], key=key.get(n, ""))
        for n in G.nodes()]).sort_values(["module", "degree"],
                                         ascending=[True, False])
    nodes.to_csv(os.path.join(TABLES, "tab_sec16_crossking_nodes.tsv"),
                 sep="\t", index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    mrows, meigs = [], {}
    for mi, c in enumerate(comps):
        c = list(c)
        if len(c) < 3:
            continue
        me = module_eigengene_net(Z, c)
        meigs["M%d" % (mi + 1)] = me
        for mcol in ["ME" + n for n in NAMES]:
            r, p = spearmanr(me, eig[mcol])
            mrows.append(dict(net_module="M%d" % (mi + 1),
                              meta_module=mcol, n=len(me), rho=r, p=p))
    mdf = pd.DataFrame(mrows)
    if len(mdf):
        _, fdr, _, _ = multipletests(mdf.p, method="fdr_bh")
        mdf["FDR"] = fdr
    mdf.to_csv(os.path.join(TABLES, "tab_sec16_netmodule_meta.tsv"),
               sep="\t", index=False)

    print(topo.to_string(index=False))
    print('\n\u5173\u952e\u5c5e\u5728\u7f51\u7edc\u4e2d\uff1a')
    print(nodes[nodes.key != ""].head(20).to_string(index=False))
    print('\n\u7f51\u7edc\u6a21\u5757 x \u4ee3\u8c22\u6a21\u5757\uff08P<0.05\uff09\uff1a')
    print(mdf[mdf.p < 0.05].sort_values("p").head(20).round(3)
          .to_string(index=False))

    # Analysis note: see README and original laboratory scripts for methodological details.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def panel(ax, letter):
        ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")

    # Analysis note: see README and original laboratory scripts for methodological details.
    big = [n for n in G.nodes()]
    subG = G.subgraph(big)
    pos = nx.spring_layout(subG, seed=7, k=0.9)
    cmap = plt.get_cmap("tab20")
    mod_ids = sorted(set(mod_of.values()),
                     key=lambda x: int(x[1:]))
    mcol = {m: cmap(i % 20) for i, m in enumerate(mod_ids)}
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for a, b, d in G.edges(data=True):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#E64B35" if d["sign"] < 0 else "#9aa4b2",
                lw=0.5, alpha=0.55,
                zorder=1)
    RING = [("keystone", "black"), ("hub", "#DAA520"),
            ("indicator", "#009E4D")]
    for n in G.nodes():
        mk = "o" if n.startswith("B|") else "^"
        col = mcol.get(mod_of.get(n), "#cccccc")
        sz = 30 + 6 * min(deg[n], 20)
        ax.scatter(pos[n][0], pos[n][1], s=sz, marker=mk, c=[col],
                   edgecolor="white", linewidth=0.3, zorder=2)
        tags = key.get(n, "").split("+") if n in key else []
        for ti, (t, c) in enumerate([rc for rc in RING if rc[0] in tags]):
            ax.scatter(pos[n][0], pos[n][1], s=sz * (1.0 + 0.6 * (ti + 1)),
                       marker=mk, facecolors="none", edgecolors=c,
                       linewidth=1.0, zorder=3)
    thr = sorted(deg.values())[-12]
    cand = sorted(G.nodes(), key=lambda n: deg[n], reverse=True)
    placed = []
    for n in cand:
        if not (n in key or deg[n] >= thr):
            continue
        x, y = pos[n]
        if all((x - px) ** 2 + (y - py) ** 2 > 0.17 ** 2 for px, py in placed):
            placed.append((x, y))
            ax.annotate(n.split("|", 1)[1][:22], (x, y), fontsize=5.6,
                        xytext=(4, 3), textcoords="offset points")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                      markersize=6, label="Bacteria"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                      markersize=6, label="Fungi"),
               Line2D([0], [0], color="#9aa4b2", lw=1, label="Positive edge"),
               Line2D([0], [0], color="#E64B35", lw=1, label="Negative edge")]
    handles += [Line2D([0], [0], marker="^", color="w", markerfacecolor="none",
                       markeredgecolor=c, markersize=8, label=t.capitalize())
                for t, c in RING]
    handles += [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=mcol[m], markersize=6,
                       label="%s (B%d/F%d)" % (
                           m, sum(1 for n in G.nodes()
                                  if mod_of.get(n) == m and
                                  n.startswith("B|")),
                           sum(1 for n in G.nodes()
                               if mod_of.get(n) == m and
                               not n.startswith("B|"))))
                for m in mod_ids if m in mcol and
                sum(1 for n in G.nodes() if mod_of.get(n) == m) >= 10]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.set_title("Fruit cross-kingdom co-occurrence network (QMP; "
                 "|r|>0.6, P<0.001)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "g")
    out = os.path.join(FIGS, "fig_sec16_g.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("->", out)

    # Analysis note: see README and original laboratory scripts for methodological details.
    gephi_dir = os.path.join(os.path.dirname(FIGS), "gephi")
    os.makedirs(gephi_dir, exist_ok=True)
    core = pd.read_csv(os.path.join(TABLES, "tab_sec16_core_taxa.tsv"),
                       sep="\t")
    cmap = {(r.kingdom, r.genus): (bool(r.core), r.prevalence,
                                    r.mean_qmp, r.strategy)
            for r in core.itertuples()}

    def _rgba_rgb(t):
        return tuple(int(round(v * 255)) for v in t[:3])

    for n in G.nodes():
        kingdom = "bacteria" if n.startswith("B|") else "fungi"
        genus = n.split("|", 1)[1]
        cflag, prev, mqmp, strat = cmap.get((kingdom, genus),
                                             (False, np.nan, np.nan, ""))
        col_tup = mcol.get(mod_of.get(n), (0.8, 0.8, 0.8, 1.0))
        rr, gg, bb = _rgba_rgb(col_tup)
        G.nodes[n]["label"] = genus
        G.nodes[n]["kingdom"] = kingdom
        G.nodes[n]["genus"] = genus
        G.nodes[n]["module"] = mod_of.get(n, "singleton")
        G.nodes[n]["degree"] = int(deg[n])
        G.nodes[n]["key_type"] = key.get(n, "")
        G.nodes[n]["core"] = "yes" if cflag else "no"
        G.nodes[n]["prevalence"] = float(prev) if prev == prev else 0.0
        G.nodes[n]["mean_qmp"] = float(mqmp) if mqmp == mqmp else 0.0
        G.nodes[n]["strategy"] = strat
        G.nodes[n]["viz"] = {
            "position": {"x": float(pos[n][0]), "y": float(pos[n][1]),
                         "z": 0.0},
            "color": {"r": rr, "g": gg, "b": bb, "a": 1.0},
            "size": float(10 + 2 * min(deg[n], 20))}
    gexf_path = os.path.join(gephi_dir, "sec16_crossking_network.gexf")
    nx.write_gexf(G, gexf_path)
    print("->", gexf_path)

    # Analysis note: see README and original laboratory scripts for methodological details.
    if len(meigs):
        M = pd.DataFrame(meigs).T
        rho = np.zeros((len(M), len(NAMES)))
        pmat = np.ones_like(rho)
        for i, m in enumerate(M.index):
            for j, n in enumerate(NAMES):
                r, p = spearmanr(M.loc[m], eig["ME" + n])
                rho[i, j] = r
                pmat[i, j] = p
        labs = []
        for m in M.index:
            mem = [n for n in G.nodes() if mod_of.get(n) == m]
            nb = sum(1 for n in mem if n.startswith("B|"))
            labs.append("%s (B%d/F%d)" % (m, nb, len(mem) - nb))
        fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(M) + 1.6))
        base = np.where(pmat < 0.05, rho, np.nan)
        im = ax.imshow(base, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.imshow(np.where(np.isnan(base), 0.93, np.nan), aspect="auto",
                  cmap="Greys", vmin=0, vmax=1, alpha=0.5)
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels(["ME" + n for n in NAMES], fontsize=6.5,
                           rotation=45, ha="right")
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(labs, fontsize=7)
        fdr = mdf.pivot_table(index="net_module", columns="meta_module",
                              values="FDR").reindex(
            index=M.index, columns=["ME" + n for n in NAMES])
        for i in range(len(M)):
            for j in range(len(NAMES)):
                if pmat[i, j] < 0.05:
                    star = "*" if fdr.values[i, j] < 0.05 else ""
                    ax.text(j, i, "%.2f%s" % (rho[i, j], star),
                            ha="center", va="center", fontsize=5.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Spearman rho", size=7)
        ax.set_title("Network modules x metabolite modules "
                     "(grey = P>=0.05; *FDR<0.05)", fontsize=8)
        panel(ax, "h")
        out = os.path.join(FIGS, "fig_sec16_h.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("->", out)


if __name__ == "__main__":
    sys.exit(main())
