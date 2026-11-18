# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from analysis_configuration import ROOT, REG_MAP

RNG = np.random.default_rng(42)

# Analysis note: see README and original laboratory scripts for methodological details.
# Analysis note: see README and original laboratory scripts for methodological details.
TAX_COLS = ["Taxonomy", "superkingdom", "phylum", "class", "order",
            "family", "genus", "species"]


# ------------------------------------------------------------------ load ----
def load_dataset(ds="16S"):
    """Research analysis helper. See README for pipeline and methodology."""
    def _split(p):
        df = pd.read_csv(p, sep="\t", index_col=0)
        tc = [c for c in TAX_COLS if c in df.columns]
        return df.drop(columns=tc).astype(np.float32), df[tc]

    abss, tax = _split(os.path.join(ROOT, ds, "unit_sample_otu_copies.xls"))
    rel, _ = _split(os.path.join(ROOT, ds, "otu.xls"))
    rel = rel.reindex(index=abss.index, columns=abss.columns).fillna(0.0)

    # Analysis note: see README and original laboratory scripts for methodological details.
    # Analysis note: see README and original laboratory scripts for methodological details.
    abss.columns = ["SX_S_8" if c == "X_S_8" else c for c in abss.columns]
    rel.columns = abss.columns

    meta = pd.read_csv(os.path.join(ROOT, "result", "r_in", "sample_meta.tsv"),
                       sep="\t")
    meta["region"] = meta["region"].replace(REG_MAP)
    # Analysis note: see README and original laboratory scripts for methodological details.
    if "replicate" not in meta.columns:
        meta["replicate"] = meta["sample_id"].str.extract(r"_(\d+)$")[0].astype(int)
    if "underground" not in meta.columns:
        meta["underground"] = meta["compartment"].isin(["BS", "RS", "R"])
    if "plant" not in meta.columns:
        meta["plant"] = meta["sample_id"]
    meta = meta.set_index("sample_id")
    return {"asv_abs": abss, "asv_rel": rel, "tax": tax, "meta": meta}


# ------------------------------------------------------------------- tax ----
def agg_tax(df, tax, level="genus"):
    """Research analysis helper. See README for pipeline and methodology."""
    lvl = tax[level].fillna("unclassified").astype(str)
    lvl = lvl.replace({"": "unclassified"})
    # Analysis note: see README and original laboratory scripts for methodological details.
    lvl = lvl.str.replace("_", " ")
    return df.groupby(lvl.reindex(df.index)).sum()


# -------------------------------------------------------------- distance ----
def bc_distance(df):
    '\u6837\u672c x \u7279\u5f81 DataFrame -> Bray-Curtis \u8ddd\u79bb\u65b9\u9635\u3002'
    return squareform(pdist(df.values, metric="braycurtis"))


def mantel(d1, d2, nperm=999, method="pearson"):
    """Research analysis helper. See README for pipeline and methodology."""
    iu = np.triu_indices(d1.shape[0], k=1)
    x, y = d1[iu], d2[iu]
    if method == "spearman":
        r = spearmanr(x, y)[0]

        def corr(a, b):
            return spearmanr(a, b)[0]
    else:
        r = np.corrcoef(x, y)[0, 1]

        def corr(a, b):
            return np.corrcoef(a, b)[0, 1]
    cnt = 0
    for _ in range(nperm):
        yp = y[RNG.permutation(len(y))]
        if corr(x, yp) >= r:
            cnt += 1
    return r, (cnt + 1) / (nperm + 1)


# ----------------------------------------------------------------- stats ----
def bh_q(pvals):
    'Benjamini-Hochberg FDR \u6821\u6b63\uff08\u4e0e\u65e7\u811a\u672c argsort+ranks \u5b9e\u73b0\u4e00\u81f4\uff09\u3002'
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        if q_sorted[i + 1] < q_sorted[i]:
            q_sorted[i] = q_sorted[i + 1]
    out = np.empty(n)
    out[order] = q_sorted
    return np.minimum(out, 1.0)


def sig_label(p, thr=0.05):
    '\u663e\u8457\u6027\u661f\u53f7\u6807\u7b7e\u3002'
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < thr else "ns"


# --------------------------------------------------------------- plotting ----
def set_xticks_labels(ax, ticks, labels, **kw):
    '\u4e24\u6b65\u5f0f\u523b\u5ea6\u6807\u7b7e\u8bbe\u7f6e\uff08matplotlib 3.3.2 \u4e0d\u652f\u6301 set_xticks \u53cc\u53c2\u5f62\u5f0f\uff09\u3002'
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, **kw)


def set_yticks_labels(ax, ticks, labels, **kw):
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, **kw)


# ------------------------------------------------------------------ misc ----
def month_from_row(s, fallback_i):
    '\u4ece\u5b57\u7b26\u4e32\u4e2d\u62bd\u53d6\u6708\u4efd\uff08\u6b63\u5219 \\d{4}-(\\d{2})\uff09\uff0c\u5931\u8d25\u56de\u9000\u884c\u5e8f\u3002'
    m = re.search(r"\d{4}-(\d{2})", str(s))
    if m:
        return int(m.group(1))
    return fallback_i
