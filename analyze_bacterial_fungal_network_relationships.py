# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare, mannwhitneyu

warnings.simplefilter("ignore")

import analysis_helpers
from analysis_configuration import ROOT, COMP_ORDER, TABLES
from infer_microbiome_cooccurrence_networks import AG, BG, PREV_F, build_graph, clr_transform, \
    infer_network

GEPHI = os.path.join(ROOT, "result", "gephi")
TOPN = 80   # Analysis note: see README and original laboratory scripts for methodological details.
STRATS = ["Colonizer", "Transmitter", "Transient"]
TAG = {"Colonizer": "C", "Transmitter": "T", "Transient": "X"}


def genus_table(ds, meta):
    dat = analysis_helpers.load_dataset(ds)
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")
    return G.loc[:, meta.index.intersection(G.columns)]


def subset_keep(G, meta, comps):
    sel = meta.index[meta["compartment"].isin(comps)]
    X = G[sel].T
    prev = (X > 0).mean(axis=0)
    keep = prev[prev >= PREV_F].index
    return X, X[keep].mean().sort_values(ascending=False).index


def mw2p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    u, p1 = mannwhitneyu(a, b)
    return min(1.0, 2 * min(p1, 1 - p1))


def main():
    meta = analysis_helpers.load_dataset("16S")["meta"]      # Analysis note: see README and original laboratory scripts for methodological details.
    Gb, Gf = genus_table("16S", meta), genus_table("ITS", meta)
    strat = {}
    for ds, key in [("16S", "b"), ("ITS", "f")]:
        strat[key] = pd.read_csv(
            os.path.join(TABLES, "tab_sec15_strategy_%s.tsv" % ds),
            sep="\t").set_index("genus")["strategy"].to_dict()

    topo_rows, edge_rows, node_rows = [], [], []
    scopes = [(c, [c]) for c in COMP_ORDER] + [("AG", AG), ("BG", BG)]
    for name, comps in scopes:
        Xb, kb = subset_keep(Gb, meta, comps)
        Xf, kf = subset_keep(Gf, meta, comps)
        kb, kf = kb[:TOPN], kf[:TOPN]
        if len(kb) < 10 or len(kf) < 10:
            continue
        Z = np.hstack([clr_transform(Xb[kb].values),
                       clr_transform(Xf[kf].values)])
        part = infer_network(Z)
        # Analysis note: see README and original laboratory scripts for methodological details.
        kb_p = ["B|" + n for n in kb]
        kf_p = ["F|" + n for n in kf]
        names = kb_p + kf_p
        g = build_graph(part, names)
        bset = set(kb_p)
        is_b = {n: n in bset for n in g.nodes()}
        lab = {n: n[2:] for n in names}

        ws = [d["weight"] for _, _, d in g.edges(data=True)]
        cross_e = [(a, b, d) for a, b, d in g.edges(data=True)
                   if is_b[a] != is_b[b]]
        cws = [d["weight"] for _, _, d in cross_e]
        topo_rows.append(dict(
            scope=name, n_bact=len(kb), n_fungi=len(kf),
            n_edges=g.number_of_edges(), n_cross=len(cross_e),
            cross_frac=len(cross_e) / g.number_of_edges()
            if g.number_of_edges() else np.nan,
            cross_pos_ratio=sum(w > 0 for w in cws) / len(cws)
            if cws else np.nan,
            all_pos_ratio=sum(w > 0 for w in ws) / len(ws) if ws else np.nan))

        # Analysis note: see README and original laboratory scripts for methodological details.
        sb = {n: strat["b"].get(lab[n], "Transient") for n in kb_p}
        sf = {n: strat["f"].get(lab[n], "Transient") for n in kf_p}
        obs = np.zeros((3, 3), int)
        for a, b, d in cross_e:
            ib, if_ = (sb[a], sf[b]) if is_b[a] else (sb[b], sf[a])
            obs[STRATS.index(ib), STRATS.index(if_)] += 1
        # Analysis note: see README and original laboratory scripts for methodological details.
        fb = pd.Series(list(sb.values())).value_counts(normalize=True) \
            .reindex(STRATS).fillna(0).values
        ff = pd.Series(list(sf.values())).value_counts(normalize=True) \
            .reindex(STRATS).fillna(0).values
        exp = np.outer(fb, ff) * obs.sum()
        # Analysis note: see README and original laboratory scripts for methodological details.
        try:
            p_f = chisquare(obs.sum(axis=0), ff * obs.sum())[1]
        except ValueError:
            p_f = np.nan
        try:
            p_b = chisquare(obs.sum(axis=1), fb * obs.sum())[1]
        except ValueError:
            p_b = np.nan
        for i, s1 in enumerate(STRATS):
            for j, s2 in enumerate(STRATS):
                e = exp[i, j]
                edge_rows.append(dict(
                    scope=name, bact_strategy=s1, fungi_strategy=s2,
                    n_cross_edges=int(obs[i, j]),
                    expected=round(float(e), 2) if e == e else np.nan,
                    OE_ratio=float(obs[i, j] / e) if (e == e and e > 0)
                    else np.nan))
        # Analysis note: see README and original laboratory scripts for methodological details.
        topo_rows[-1]["gof_bact_p"] = p_b
        topo_rows[-1]["gof_fungi_p"] = p_f
        chi2, p_ind = np.nan, p_f

        # Analysis note: see README and original laboratory scripts for methodological details.
        cd = pd.Series(0, index=names)
        for a, b, d in cross_e:
            cd[a] += 1
            cd[b] += 1
        for key, ds, smap, pool in [("bact", "16S", sb, kb_p),
                                    ("fungi", "ITS", sf, kf_p)]:
            for s in STRATS:
                deg = cd[[n for n in pool if smap[n] == s]]
                node_rows.append(dict(
                    scope=name, kingdom=ds, strategy=s, n_nodes=len(deg),
                    median_cross_degree=float(deg.median())
                    if len(deg) else np.nan))
            dc = cd[[n for n in pool if smap[n] == "Colonizer"]]
            dt = cd[[n for n in pool if smap[n] == "Transient"]]
            node_rows.append(dict(
                scope=name, kingdom=ds, strategy="MW_p_C_vs_T",
                n_nodes=np.nan, median_cross_degree=mw2p(dc, dt)))

        # Analysis note: see README and original laboratory scripts for methodological details.
        pd.DataFrame([dict(
            Id=n, Label=lab[n], Kingdom="16S" if is_b[n] else "ITS",
            Strategy=(sb if is_b[n] else sf)[n],
            Degree=int(g.degree(n)), CrossDegree=int(cd[n]))
            for n in g.nodes()]).to_csv(
            os.path.join(GEPHI, "CK_%s_nodes.csv" % name), index=False)
        pd.DataFrame([dict(
            Source=a, Target=b, Weight=round(d["weight"], 4),
            Sign="positive" if d["weight"] > 0 else "negative",
            Type="cross" if is_b[a] != is_b[b] else
            ("intra_16S" if is_b[a] else "intra_ITS"))
            for a, b, d in g.edges(data=True)]).to_csv(
            os.path.join(GEPHI, "CK_%s_edges.csv" % name), index=False)
        print("[CK] %-4s bact=%3d fungi=%3d edges=%4d cross=%3d (%.2f) "
              "pos_cross=%.2f chi2 p=%.3f" % (
                  name, len(kb), len(kf), g.number_of_edges(), len(cross_e),
                  len(cross_e) / g.number_of_edges(),
                  (sum(w > 0 for w in cws) / len(cws)) if cws else np.nan,
                  p_ind))

    pd.DataFrame(topo_rows).to_csv(
        os.path.join(TABLES, "tab_sec15_crossking_topo.tsv"),
        sep="\t", index=False)
    pd.DataFrame(edge_rows).to_csv(
        os.path.join(TABLES, "tab_sec15_crossking_edges.tsv"),
        sep="\t", index=False)
    pd.DataFrame(node_rows).to_csv(
        os.path.join(TABLES, "tab_sec15_crossking_nodecross.tsv"),
        sep="\t", index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for ds, G, key in [("16S", Gb, "b"), ("ITS", Gf, "f")]:
        df = pd.read_csv(os.path.join(
            TABLES, "tab_sec15_strategy_%s.tsv" % ds), sep="\t")
        mean_load = G.mean(axis=1)
        for s in STRATS:
            genera = df.loc[df.strategy == s, "genus"].dropna()
            genera = [g for g in genera if g in mean_load.index]
            top = sorted(genera, key=lambda g: -mean_load[g])[:50]
            rows = []
            for gn in top:
                r = dict(genus=gn, mean_load=round(float(mean_load[gn]), 1))
                for c in COMP_ORDER:
                    v = G.loc[gn][meta["compartment"] == c]
                    r[c] = round(float(np.log10(v.median() + 1)), 3)
                rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(
                TABLES, "tab_sec15_heatmap_top50_%s_%s.tsv"
                % (ds, TAG[s])), sep="\t", index=False)
            print("[HM] %s %s: top50 rows=%d" % (ds, s, len(rows)))
    print("done.")


if __name__ == "__main__":
    main()
