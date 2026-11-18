# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2_contingency

import analysis_helpers
from analysis_configuration import ROOT, COMP_ORDER, TABLES, DATASETS

UPSTREAM = {"RS": "BS", "R": "RS", "S": "R", "L": "S", "F": "L"}
PREV_T = 0.3          # Analysis note: see README and original laboratory scripts for methodological details.
PREV_TARGET = 0.5     # Analysis note: see README and original laboratory scripts for methodological details.
LOAD_TARGET = 1e4     # Analysis note: see README and original laboratory scripts for methodological details.


def genus_metrics(ds):
    dat = analysis_helpers.load_dataset(ds)
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")   # genus x sample
    meta = dat["meta"]
    G = G.loc[:, meta.index.intersection(G.columns)]
    meta = meta.loc[G.columns]

    feast = pd.read_csv(os.path.join(TABLES, "tab_eco_feast_%s.tsv" % ds),
                        sep="\t")

    rows = []
    totals = G.sum(axis=0).replace(0, np.nan)
    for g in G.index:
        load = G.loc[g]
        prev_c, med_c = {}, {}
        for c in COMP_ORDER:
            s = load[meta["compartment"] == c]
            prev_c[c] = float((s > 0).mean())
            med_c[c] = float(s.median())
        prev_overall = float((load > 0).mean())
        n_comp = sum(1 for c in COMP_ORDER if prev_c[c] >= PREV_T)
        cand = [c for c in COMP_ORDER if prev_c[c] >= 0.2]
        target = max(cand, key=lambda c: med_c[c]) if cand else \
            max(COMP_ORDER, key=lambda c: prev_c[c])
        up = UPSTREAM.get(target)
        grad = (med_c[target] / med_c[up]) if (up and med_c[up] > 0) else \
            (med_c["BS"] / med_c["RS"] if target == "BS" and med_c["RS"] > 0
             else np.nan)
        # Analysis note: see README and original laboratory scripts for methodological details.
        rel = load / totals
        best = 0.0
        for (frm, to), grp in feast.groupby(["from", "to"]):
            m = grp.set_index("sink_sample")["contribution"]
            r = rel.reindex(m.index).fillna(0.0)
            pos = r[r > 0]
            if len(pos) == 0:
                continue
            attrib = (pos * m.reindex(pos.index)).median()
            best = max(best, float(attrib))
        rows.append(dict(genus=g, target=target, prev_overall=prev_overall,
                         prev_target=prev_c[target],
                         med_load_target=med_c[target],
                         grad_ratio=grad, feast_down=best, n_comp=n_comp))
    df = pd.DataFrame(rows)

    # Analysis note: see README and original laboratory scripts for methodological details.
    nz = df["feast_down"][df["feast_down"] > 0]
    contrib_t = float(nz.median()) if len(nz) else 0.0
    cond_c = (df["prev_target"] >= PREV_TARGET) & \
             (df["med_load_target"] >= LOAD_TARGET) & \
             (df["feast_down"] >= contrib_t)
    cond_t = (~cond_c) & (df["n_comp"] >= 4)
    df["strategy"] = np.where(cond_c, "Colonizer",
                              np.where(cond_t, "Transmitter", "Transient"))
    df.attrs["contrib_t"] = contrib_t
    return df, G, meta


def comp_site_summaries(df, G, meta, ds):
    # Analysis note: see README and original laboratory scripts for methodological details.
    comp_rows = []
    for c in COMP_ORDER:
        sel = []
        for g in df["genus"]:
            s = G.loc[g][meta["compartment"] == c]
            if (s > 0).mean() >= 0.2:
                sel.append(g)
        sub = df[df["genus"].isin(sel)]
        cnt = sub["strategy"].value_counts()
        comp_rows.append(dict(compartment=c, n_genera=len(sub),
                              colonizer=cnt.get("Colonizer", 0) / len(sub),
                              transmitter=cnt.get("Transmitter", 0) / len(sub),
                              transient=cnt.get("Transient", 0) / len(sub)))
    comp = pd.DataFrame(comp_rows)

    # Analysis note: see README and original laboratory scripts for methodological details.
    site_rows, tab = [], []
    for st in ["WH", "SH", "SX"]:
        sel = [g for g in df["genus"]
               if (G.loc[g][meta["region"] == st] > 0).mean() >= 0.2]
        sub = df[df["genus"].isin(sel)]
        cnt = sub["strategy"].value_counts()
        site_rows.append(dict(site=st, n_genera=len(sub),
                              colonizer=cnt.get("Colonizer", 0) / len(sub),
                              transmitter=cnt.get("Transmitter", 0) / len(sub),
                              transient=cnt.get("Transient", 0) / len(sub)))
        tab.append([cnt.get(s, 0) for s in
                    ["Colonizer", "Transmitter", "Transient"]])
    chi2, p_site = (np.nan, np.nan)
    try:
        chi2, p_site, _, _ = chi2_contingency(np.array(tab))
    except ValueError:
        pass
    site = pd.DataFrame(site_rows)
    site.attrs["chi2_p"] = p_site
    return comp, site, p_site


def icamp_link(df, comp, ds):
    ic = pd.read_csv(os.path.join(TABLES, "tab_eco_icamp_%s.tsv" % ds),
                     sep="\t")
    det = ic[ic["Category"] == "Deterministic"] \
        .groupby("Compartment")["Proportion"].mean()
    m = comp.set_index("compartment")["colonizer"]
    idx = [c for c in COMP_ORDER if c in det.index and c in m.index]
    r, p = spearmanr(m.reindex(idx).values, det.reindex(idx).values)
    out = pd.DataFrame(dict(compartment=idx,
                            colonizer_frac=m.reindex(idx).values,
                            deterministic_frac=det.reindex(idx).values))
    hi = out[out["colonizer_frac"] >= out["colonizer_frac"].median()]
    lo = out[out["colonizer_frac"] < out["colonizer_frac"].median()]
    from scipy.stats import mannwhitneyu
    # Analysis note: see README and original laboratory scripts for methodological details.
    u_stat, u_p1 = mannwhitneyu(hi["deterministic_frac"],
                                lo["deterministic_frac"])
    u_p = min(1.0, 2 * min(u_p1, 1 - u_p1)) if len(hi) and len(lo) else np.nan
    return out, r, p, u_p


def main():
    for ds in ["16S", "ITS"]:
        df, G, meta = genus_metrics(ds)
        df.to_csv(os.path.join(TABLES, "tab_sec15_strategy_%s.tsv" % ds),
                  sep="\t", index=False)
        comp, site, p_site = comp_site_summaries(df, G, meta, ds)
        comp.to_csv(os.path.join(TABLES,
                                 "tab_sec15_strategy_comp_%s.tsv" % ds),
                    sep="\t", index=False)
        site.to_csv(os.path.join(TABLES,
                                 "tab_sec15_strategy_site_%s.tsv" % ds),
                    sep="\t", index=False)
        out, r, p, u_p = icamp_link(df, comp, ds)
        out.to_csv(os.path.join(TABLES,
                                "tab_sec15_strategy_icamp_%s.tsv" % ds),
                   sep="\t", index=False)
        n_c = (df["strategy"] == "Colonizer").sum()
        n_t = (df["strategy"] == "Transmitter").sum()
        n_x = (df["strategy"] == "Transient").sum()
        print("[%s] strategy C/T/X = %d/%d/%d; contrib_t=%.4f; site chi2 p=%.4f"
              % (ds, n_c, n_t, n_x, df.attrs["contrib_t"], p_site))
        print("  comp colonizer frac:",
              {c: round(v, 3) for c, v in
               zip(comp["compartment"], comp["colonizer"])})
        print("  icamp spearman rho=%.3f p=%.3f; hi-vs-lo MW p=%.3f"
              % (r, p, u_p))


if __name__ == "__main__":
    main()
