# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, \
    modularity
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.covariance import GraphicalLassoCV

# Analysis note: see README and original laboratory scripts for methodological details.
warnings.simplefilter("ignore")

import analysis_helpers
from analysis_configuration import ROOT, COMP_ORDER, TABLES, DATASETS

FIGS = os.path.join(ROOT, "result", "figs")
GEPHI = os.path.join(ROOT, "result", "gephi")
PREV_F = 0.3
# Analysis note: see README and original laboratory scripts for methodological details.
# Analysis note: see README and original laboratory scripts for methodological details.
MAX_TAXA = {"comp": None, "site": 30, "pool": None}
AG = ["S", "L", "F"]
BG = ["BS", "RS", "R"]
RNG = np.random.default_rng(42)

# Analysis note: see README and original laboratory scripts for methodological details.
FUNC = {
    "Streptomyces": '\u6297\u751f\u7d20/\u6b21\u7ea7\u4ee3\u8c22\u7269\u5408\u6210\uff1bPGPR',
    "Bradyrhizobium": '\u56fa\u6c2e\uff1b\u5171\u751f/\u5185\u751f',
    "Pseudomonas": 'PGPR\uff1b\u751f\u9632\uff1b\u89e3\u78f7',
    "Methylobacterium": 'C1 \u4ee3\u8c22\uff1b\u53f6\u9645 PGPR',
    "Sphingomonas": '\u82b3\u9999\u65cf\u964d\u89e3\uff1b\u6297\u9006\u4fc3\u751f',
    "Gaiella": '\u571f\u58e4\u6709\u673a\u8d28\u5468\u8f6c',
    "Bacillus": '\u82bd\u5b62\u6297\u9006\uff1b\u751f\u9632\uff1bPGPR',
    "Rhizobium": '\u56fa\u6c2e\uff1b\u5171\u751f',
    "Nitrospira": '\u785d\u5316\uff08\u4e9a\u785d\u9178\u76d0\u6c27\u5316\uff09',
    "Nitrososphaera": '\u6c28\u6c27\u5316\u53e4\u83cc',
    "Burkholderia-Caballeronia-Paraburkholderia": 'PGPR\uff1b\u751f\u9632',
    "Massilia": '\u6839\u9645\u4fc3\u751f\uff1b\u51e0\u4e01\u8d28\u964d\u89e3',
    "Flavobacterium": '\u591a\u7cd6\u964d\u89e3',
    "Chryseobacterium": '\u51e0\u4e01\u8d28/\u591a\u7cd6\u964d\u89e3',
    "Alternaria": '\u5185\u751f/\u6761\u4ef6\u81f4\u75c5\u771f\u83cc',
    "Cladosporium": '\u5185\u751f/\u8150\u751f\u771f\u83cc',
    "Fusarium": '\u571f\u4f20\u771f\u83cc\uff1b\u90e8\u5206\u81f4\u75c5',
    "Penicillium": '\u8150\u751f\uff1b\u6297\u751f\u7d20\u5408\u6210',
    "Trichoderma": '\u751f\u9632\u771f\u83cc\uff1b\u4fc3\u751f',
    "Mortierella": '\u8150\u751f\uff1b\u8102\u8d28\u4ee3\u8c22',
    "Cryptococcus": '\u53f6\u9645\u9175\u6bcd\uff1b\u6297\u9006',
    "Aureobasidium": '\u53f6\u9645\u8010\u9006\u771f\u83cc',
    "Vishniacozyma": '\u53f6\u9645\u9175\u6bcd',
    "Saitozyma": '\u53f6\u9645\u9175\u6bcd',
    "Hannaella": '\u53f6\u9645\u9175\u6bcd',
    "unclassified": '\u672a\u6ce8\u91ca',
}


def clr_transform(X):
    'samples x taxa \u7edd\u5bf9\u8f7d\u91cf -> CLR\uff08log1p \u540e\u6837\u672c\u5185\u4e2d\u5fc3\u5316\uff09\u3002'
    L = np.log1p(np.asarray(X, dtype=float))
    return L - L.mean(axis=1, keepdims=True)


def infer_network(X_clr, cv=3):
    """Research analysis helper. See README for pipeline and methodology."""
    gl = GraphicalLassoCV(cv=min(cv, max(2, X_clr.shape[0] - 1)),
                          max_iter=3000, tol=1e-4)
    gl.fit(X_clr)
    prec = gl.precision_
    d = np.sqrt(np.diag(prec))
    part = -prec / np.outer(d, d)
    np.fill_diagonal(part, 0.0)
    return np.nan_to_num(part, nan=0.0)   # Analysis note: see README and original laboratory scripts for methodological details.


def build_graph(part, names):
    g = nx.Graph()
    g.add_nodes_from(names)
    ii, jj = np.where(np.abs(part) > 1e-8)
    for a, b in zip(ii, jj):
        if a < b:
            g.add_edge(names[a], names[b], weight=float(part[a, b]))
    return g


def topo(g):
    n = g.number_of_nodes()
    e = g.number_of_edges()
    b = nx.Graph()                       # Analysis note: see README and original laboratory scripts for methodological details.
    b.add_nodes_from(g.nodes())
    b.add_edges_from(g.edges())
    out = dict(n_nodes=n, n_edges=e,
               mean_degree=2 * e / n if n else np.nan,
               density=nx.density(b) if n > 1 else np.nan,
               clustering=nx.average_clustering(b) if e else np.nan)
    comps = list(nx.connected_components(b))
    lcc = max(comps, key=len) if comps else set()
    gl = b.subgraph(lcc)
    out["apl"] = nx.average_shortest_path_length(gl) \
        if gl.number_of_nodes() > 1 else np.nan
    try:
        comms = []
        for cc in comps:                      # Analysis note: see README and original laboratory scripts for methodological details.
            sub = b.subgraph(cc)
            if sub.number_of_edges() == 0 or sub.number_of_nodes() < 3:
                comms.extend({frozenset([x]) for x in cc})
            else:
                comms.extend(greedy_modularity_communities(sub))
        out["modularity"] = modularity(b, comms)
    except Exception:
        comms, out["modularity"] = [frozenset(b.nodes())], 0.0
    ws = [d["weight"] for _, _, d in g.edges(data=True)]
    out["pos_edge_ratio"] = sum(w > 0 for w in ws) / len(ws) if ws else np.nan
    return out, comms


def zi_pi(g, comms):
    mod_of = {n: i for i, c in enumerate(comms) for n in c}
    rows = []
    for n in g.nodes():
        m = mod_of[n]
        ki = g.degree(n)
        neigh = list(g.neighbors(n))
        k_im = sum(1 for x in neigh if mod_of[x] == m)
        deg_in_m = [g.degree(x) for x in g.nodes() if mod_of[x] == m]
        mu, sd = np.mean(deg_in_m), np.std(deg_in_m)
        zi = (k_im - mu) / sd if sd > 0 else 0.0
        if ki > 0:
            cnt = {}
            for x in neigh:
                cnt[mod_of[x]] = cnt.get(mod_of[x], 0) + 1
            pi = 1 - sum((v / ki) ** 2 for v in cnt.values())
        else:
            pi = 0.0
        hub = ("Network hub" if zi > 2.5 and pi > 0.62 else
               "Module hub" if zi > 2.5 else
               "Connector hub" if pi > 0.62 else "Peripheral")
        rows.append(dict(genus=n, module=m, degree=ki, Zi=zi, Pi=pi,
                         hub_type=hub, keystone=hub != "Peripheral"))
    return pd.DataFrame(rows)


def robustness(g, keystone_set, n_rand=20, frac_max=0.5, step=0.05):
    nodes = list(g.nodes())
    n = len(nodes)
    if n < 10:
        return None
    fracs = np.round(np.arange(0, frac_max + 1e-9, step), 3)
    lcc0 = max(len(c) for c in nx.connected_components(g))

    def lcc_frac(sub):
        if sub.number_of_nodes() == 0:
            return 0.0
        return max(len(cc) for cc in nx.connected_components(sub)) / lcc0

    def curve(order):
        vals = []
        removed = set()
        for f in fracs:
            vals.append(lcc_frac(g.subgraph([x for x in nodes
                                             if x not in removed])))
            target = int(round((f + step) * n))
            while len(removed) < min(target, n):
                removed.add(order[len(removed)])
        return np.array(vals)

    order_t = sorted(nodes, key=lambda x: -g.degree(x))
    ks = [x for x in order_t if x in keystone_set]
    rest = [x for x in order_t if x not in keystone_set]
    order_k = ks + rest
    rnd = np.array([curve(list(RNG.permutation(nodes)))
                    for _ in range(n_rand)])
    return pd.DataFrame(dict(frac=fracs,
                             random_mean=rnd.mean(axis=0),
                             random_sd=rnd.std(axis=0),
                             targeted=curve(order_t),
                             keystone_first=curve(order_k)))


def genus_table(ds, meta):
    dat = analysis_helpers.load_dataset(ds)
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")
    G = G.loc[:, meta.index.intersection(G.columns)]
    return G


def subset(G, meta, comps=None, site=None):
    sel = meta.index
    if site:
        sel = sel[meta.loc[sel, "region"] == site]
    if comps:
        sel = sel[meta.loc[sel, "compartment"].isin(comps)]
    X = G[sel].T                       # samples x genera
    prev = (X > 0).mean(axis=0)
    keep = prev[prev >= PREV_F].index
    keep = X[keep].mean().sort_values(ascending=False).index
    return X, keep


def main():
    os.makedirs(GEPHI, exist_ok=True)
    for ds in ["16S", "ITS"]:
        meta = analysis_helpers.load_dataset(ds)["meta"]
        G = genus_table(ds, meta)
        strat = pd.read_csv(os.path.join(
            TABLES, "tab_sec15_strategy_%s.tsv" % ds), sep="\t") \
            .set_index("genus")["strategy"].to_dict()

        topo_rows, key_rows, rob_blocks, graph_store = [], [], [], {}
        scopes = [("comp", c, None, [c]) for c in COMP_ORDER] + \
                 [("site", "%s.%s" % (c, st), st, [c])
                  for c in COMP_ORDER for st in ["WH", "SH", "SX"]] + \
                 [("pool", "AG", None, AG), ("pool", "BG", None, BG)]

        for kind, name, site, comps in scopes:
            X, keep = subset(G, meta, comps, site)
            keep = keep[:MAX_TAXA[kind]]
            if len(keep) < 10:
                continue
            part = infer_network(clr_transform(X[keep].values))
            g = build_graph(part, list(keep))
            t, comms = topo(g)
            topo_rows.append(dict(scope_kind=kind, scope=name, site=site,
                                  n_samples=X.shape[0], n_taxa=len(keep),
                                  **t))
            graph_store[(kind, name)] = (g, comms)
            print("[%s] net %-10s n=%3d taxa=%3d edges=%4d mod=%.3f pos=%.2f"
                  % (ds, name, X.shape[0], len(keep), g.number_of_edges(),
                     t["modularity"], t["pos_edge_ratio"]))

        topo_df = pd.DataFrame(topo_rows)
        topo_df.to_csv(os.path.join(TABLES,
                                    "tab_sec15_net_topo_%s.tsv" % ds),
                       sep="\t", index=False)

        # Analysis note: see README and original laboratory scripts for methodological details.
        ss = topo_df[topo_df["scope_kind"] == "site"]
        ss = ss.assign(group=ss["scope"].str.split(".").str[0]
                       .map(lambda c: "AG" if c in AG else "BG"))
        for par in ["n_edges", "modularity", "pos_edge_ratio"]:
            a, b = ss[ss.group == "AG"][par], ss[ss.group == "BG"][par]
            u, p1 = mannwhitneyu(a, b)
            print("[%s] AG-vs-BG %s: %.3f vs %.3f, MW p=%.4f"
                  % (ds, par, a.mean(), b.mean(),
                     min(1.0, 2 * min(p1, 1 - p1))))

        # Analysis note: see README and original laboratory scripts for methodological details.
        ind = pd.read_csv(os.path.join(
            TABLES, "tab_12_genus_specificity.tsv"), sep="\t")
        ind = ind[ind.dataset == ds]
        ind_map = dict(zip(ind.genus, ind.enriched_compartment))
        cons_rows = []
        for (kind, name), (g, comms) in graph_store.items():
            if kind not in ("comp", "pool"):
                continue
            zp = zi_pi(g, comms)
            zp["scope"] = name
            zp["function"] = zp["genus"].map(
                lambda x: FUNC.get(x, '\u672a\u6ce8\u91ca\uff08\u6587\u732e\u5f85\u8865\uff09'))
            zp["indval_compartment"] = zp["genus"].map(ind_map)
            zp["hub_indval_match"] = zp.apply(
                lambda r: (name == r["indval_compartment"])
                if pd.notnull(r["indval_compartment"]) else False, axis=1)
            zp["strategy"] = zp["genus"].map(strat)
            key_rows.append(zp)
            # Analysis note: see README and original laboratory scripts for methodological details.
            kset = set(zp.loc[zp.keystone, "genus"])
            rc = robustness(g, kset)
            if rc is not None:
                rc["scope"] = name
                frac = rc["frac"].values
                for col in ["random_mean", "targeted", "keystone_first"]:
                    rc.loc[:, "auc_" + col] = np.trapz(rc[col], frac)
                rob_blocks.append(rc)

        key_df = pd.concat(key_rows, ignore_index=True)
        key_df.to_csv(os.path.join(TABLES,
                                   "tab_sec15_keystone_%s.tsv" % ds),
                      sep="\t", index=False)

        # Analysis note: see README and original laboratory scripts for methodological details.
        for c in COMP_ORDER:
            cnt = {}
            for st in ["WH", "SH", "SX"]:
                kk = ("site", "%s.%s" % (c, st))
                if kk not in graph_store:
                    continue
                g2, comms2 = graph_store[kk]
                zp2 = zi_pi(g2, comms2)
                for x in zp2.loc[zp2.keystone, "genus"]:
                    cnt.setdefault(x, set()).add(st)
            for x, sts in cnt.items():
                cons_rows.append(dict(compartment=c, genus=x,
                                      n_sites=len(sts),
                                      sites=",".join(sorted(sts))))
        cons_df = pd.DataFrame(cons_rows)
        cons_df.to_csv(os.path.join(
            TABLES, "tab_sec15_keystone_cons_%s.tsv" % ds),
            sep="\t", index=False)
        n_cons = (cons_df.n_sites >= 2).sum() if len(cons_df) else 0
        print("[%s] keystones: %d (conserved>=2 sites: %d)"
              % (ds, key_df.keystone.sum(), n_cons))

        rob = pd.concat(rob_blocks, ignore_index=True)
        rob.to_csv(os.path.join(TABLES, "tab_sec15_robust_%s.tsv" % ds),
                   sep="\t", index=False)

        # Analysis note: see README and original laboratory scripts for methodological details.
        ic = pd.read_csv(os.path.join(TABLES, "tab_eco_icamp_%s.tsv" % ds),
                         sep="\t")
        det = ic[ic.Category == "Deterministic"].groupby("Compartment") \
            ["Proportion"].mean()
        cf = pd.read_csv(os.path.join(
            TABLES, "tab_sec15_strategy_comp_%s.tsv" % ds), sep="\t") \
            .set_index("compartment")["colonizer"]
        auc_t = rob[rob["scope"].isin(COMP_ORDER)].drop_duplicates("scope") \
            .set_index("scope")["auc_targeted"]
        integ = pd.DataFrame(dict(
            colonizer_frac=cf, deterministic_frac=det,
            robustness_auc_targeted=auc_t)).dropna()
        ks_cnt = key_df[key_df.keystone].groupby("scope").size()
        integ["n_keystones"] = ks_cnt.reindex(integ.index).fillna(0)
        if len(integ) >= 4:
            r1, p1 = spearmanr(integ["colonizer_frac"],
                               integ["robustness_auc_targeted"])
            r2, p2 = spearmanr(integ["deterministic_frac"],
                               integ["robustness_auc_targeted"])
            integ.attrs["spearman"] = (r1, p1, r2, p2)
            print("[%s] integ: colonizer-vs-robust rho=%.3f p=%.3f; "
                  "det-vs-robust rho=%.3f p=%.3f" % (ds, r1, p1, r2, p2))
        integ.to_csv(os.path.join(TABLES,
                                  "tab_sec15_integration_%s.tsv" % ds),
                     sep="\t")

        # Analysis note: see README and original laboratory scripts for methodological details.
        for (kind, name), (g, comms) in graph_store.items():
            if kind not in ("comp", "pool"):
                continue
            zp = zi_pi(g, comms).set_index("genus")
            node_rows = []
            for n in g.nodes():
                node_rows.append(dict(
                    Id=n, Label=n, Scope=name, Kingdom=DATASETS[ds],
                    Module=int(zp.loc[n, "module"]),
                    Degree=int(zp.loc[n, "degree"]),
                    Zi=round(float(zp.loc[n, "Zi"]), 3),
                    Pi=round(float(zp.loc[n, "Pi"]), 3),
                    HubType=zp.loc[n, "hub_type"],
                    Keystone=bool(zp.loc[n, "keystone"]),
                    Strategy=strat.get(n, "NA")))
            pd.DataFrame(node_rows).to_csv(
                os.path.join(GEPHI, "%s_%s_nodes.csv" % (ds, name)),
                index=False)
            pd.DataFrame([dict(Source=a, Target=b,
                               Weight=round(d["weight"], 4),
                               Sign="positive" if d["weight"] > 0
                               else "negative")
                          for a, b, d in g.edges(data=True)]).to_csv(
                os.path.join(GEPHI, "%s_%s_edges.csv" % (ds, name)),
                index=False)
            gx = nx.Graph()
            for r in node_rows:
                gx.add_node(r["Id"], **{k: r[k] for k in
                                        ["Label", "Scope", "Module",
                                         "Degree", "Zi", "Pi", "HubType",
                                         "Keystone", "Strategy", "Kingdom"]})
            for a, b, d in g.edges(data=True):
                gx.add_edge(a, b, Weight=round(d["weight"], 4),
                            Sign="positive" if d["weight"] > 0
                            else "negative")
            nx.write_gexf(gx, os.path.join(GEPHI, "%s_%s.gexf" % (ds, name)))
        print("[%s] done." % ds)


if __name__ == "__main__":
    main()
