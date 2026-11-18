# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from analysis_configuration import ROOT, COMP_ORDER, TABLES

GEPHI = os.path.join(ROOT, "result", "gephi")


def main():
    for ds in ["16S", "ITS"]:
        edges = {c: pd.read_csv(os.path.join(
            GEPHI, "%s_%s_edges.csv" % (ds, c))) for c in COMP_ORDER}

        # Analysis note: see README and original laboratory scripts for methodological details.
        cnt = {}
        for c, e in edges.items():
            for a, b in zip(e["Source"], e["Target"]):
                k = tuple(sorted((str(a), str(b))))
                cnt[k] = cnt.get(k, 0) + 1

        gs_rows = []
        for c in COMP_ORDER:
            e = edges[c]
            keys = [tuple(sorted((str(a), str(b))))
                    for a, b in zip(e["Source"], e["Target"])]
            ng = sum(1 for k in set(keys) if cnt[k] >= 2)
            gs_rows.append(dict(scope=c, n_edges=len(set(keys)),
                                n_generalist=ng,
                                prop_generalist=ng / len(set(keys))
                                if len(set(keys)) else np.nan))
        gs = pd.DataFrame(gs_rows)
        gs.to_csv(os.path.join(TABLES,
                               "tab_sec15_net_genspec_%s.tsv" % ds),
                  sep="\t", index=False)
        print("[%s] generalist edge prop: %s" % (
            ds, {r.scope: round(r.prop_generalist, 3)
                 for r in gs.itertuples()}))

        node_rows = []
        for c in COMP_ORDER:
            n = pd.read_csv(os.path.join(GEPHI, "%s_%s_nodes.csv" % (ds, c)))
            e = edges[c]
            pos = e[e["Sign"] == "positive"]
            neg = e[e["Sign"] == "negative"]
            pc = pd.concat([pos["Source"], pos["Target"]]).value_counts()
            nc = pd.concat([neg["Source"], neg["Target"]]).value_counts()
            for _, r in n.iterrows():
                p, q = int(pc.get(r["Id"], 0)), int(nc.get(r["Id"], 0))
                node_rows.append(dict(
                    scope=c, genus=r["Id"], degree=int(r["Degree"]),
                    n_pos=p, n_neg=q,
                    log10_neg_pos=float(np.log10((q + 1) / (p + 1))),
                    keystone=bool(r["Keystone"])))
        pd.DataFrame(node_rows).to_csv(
            os.path.join(TABLES, "tab_sec15_net_nodestat_%s.tsv" % ds),
            sep="\t", index=False)
        print("[%s] nodestat rows: %d" % (ds, len(node_rows)))

        strategy_link(ds)


def strategy_link(ds):
    """Research analysis helper. See README for pipeline and methodology."""
    rows = []
    for scope in list(COMP_ORDER) + ["AG", "BG"]:
        nf = os.path.join(GEPHI, "%s_%s_nodes.csv" % (ds, scope))
        ef = os.path.join(GEPHI, "%s_%s_edges.csv" % (ds, scope))
        if not (os.path.exists(nf) and os.path.exists(ef)):
            continue
        nodes = pd.read_csv(nf)
        edges = pd.read_csv(ef)
        strat = nodes.set_index("Id")["Strategy"].to_dict()
        vc = nodes["Strategy"].value_counts(normalize=True)
        deg_c = nodes.loc[nodes.Strategy == "Colonizer", "Degree"]
        deg_t = nodes.loc[nodes.Strategy == "Transient", "Degree"]
        if len(deg_c) >= 2 and len(deg_t) >= 2:
            u, p1 = mannwhitneyu(deg_c, deg_t)
            mw_p = min(1.0, 2 * min(p1, 1 - p1))
        else:
            mw_p = np.nan
        # keystone x colonizer Fisher
        ks = nodes["Keystone"].astype(bool)
        col = nodes["Strategy"] == "Colonizer"
        tab = [[int((ks & col).sum()), int((ks & ~col).sum())],
               [int((~ks & col).sum()), int((~ks & ~col).sum())]]
        try:
            or_, fp = fisher_exact(tab)
        except ValueError:
            or_, fp = np.nan, np.nan
        # Analysis note: see README and original laboratory scripts for methodological details.
        pair = [tuple(sorted((strat.get(a, "Transient"),
                              strat.get(b, "Transient"))))
                for a, b in zip(edges["Source"], edges["Target"])]
        ps = pd.Series(pair).value_counts(normalize=True)
        rows.append(dict(
            scope=scope, n_nodes=len(nodes),
            colonizer_frac=vc.get("Colonizer", 0.0),
            transmitter_frac=vc.get("Transmitter", 0.0),
            transient_frac=vc.get("Transient", 0.0),
            med_deg_colonizer=float(deg_c.median()) if len(deg_c) else np.nan,
            med_deg_transient=float(deg_t.median()) if len(deg_t) else np.nan,
            mw_p_deg_C_vs_T=mw_p,
            edge_CC_frac=float(ps.get(("Colonizer", "Colonizer"), 0.0)),
            edge_TT_frac=float(ps.get(("Transient", "Transient"), 0.0)),
            ks_colonizer_OR=or_, ks_colonizer_p=fp))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "tab_sec15_net_strategy_%s.tsv" % ds),
               sep="\t", index=False)
    print("[%s] strategy-link: colonizer node frac %s; keystone-colonizer "
          "Fisher p %s" % (
              ds, {r.scope: round(r.colonizer_frac, 2)
                   for r in out.itertuples()},
              {r.scope: round(r.ks_colonizer_p, 3)
               for r in out.itertuples()}))


if __name__ == "__main__":
    main()
