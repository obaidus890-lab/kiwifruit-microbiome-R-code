# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.simplefilter("ignore")

from analysis_configuration import TABLES
from analyze_fruit_metabolome_microbiome_associations import MET_TO_REG, load_metabolome, sample_map

META_DIR = os.path.abspath(os.environ.get(
    "KIWI_METABOLOMICS_DIR", os.path.join(os.environ.get("KIWI_ROOT", os.path.dirname(__file__)), "metabolomics")))


def module_eigengenes():
    '\u8fd4\u56de DataFrame\uff1a\u6837\u672c x \u6a21\u5757 eigengene\uff0827 x n_mod\uff09\u3002'
    met, groups, fmap = sample_map()
    met = met.T
    assign = pd.read_csv(os.path.join(
        META_DIR, "analysis_output", "WGCNA",
        "WGCNA_module_assignment.csv"))
    col_met, col_mod = assign.columns[0], [c for c in assign.columns
                                           if "module" in c.lower()][0]
    z = (met - met.mean()) / met.std(ddof=1)
    z = z.fillna(0)
    eig = {}
    for mod, sub in assign.groupby(col_mod):
        feats = [m for m in sub[col_met] if m in z.columns]
        if len(feats) < 5:
            continue
        M = z[feats].values
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        sign = 1 if vt[0].sum() >= 0 else -1
        eig["ME%s" % mod] = u[:, 0] * s[0] * sign
    return pd.DataFrame(eig, index=met.index), met, fmap


def main():
    eig, met, fmap = module_eigengenes()
    inv_fmap = {v: k for k, v in fmap.items()}
    pref2reg = {"WH": "WH", "BZ": "SH", "SX": "SX"}
    reg = pd.Series({met_s: pref2reg[mic_s.split("_")[0]]
                     for mic_s, met_s in inv_fmap.items()})

    rows = []
    for ds in ["16S", "ITS"]:
        f = pd.read_csv(os.path.join(TABLES,
                                     "tab_eco_feast_%s.tsv" % ds), sep="\t")
        lf = f[(f["from"] == "L") & (f["to"] == "F")].copy()
        lf["met_sample"] = lf["sink_sample"].map(inv_fmap)
        lf = lf.dropna(subset=["met_sample"])
        contrib = lf.set_index("met_sample")["contribution"]
        common_idx = contrib.index.intersection(eig.index)
        for scope, idx in [("pooled", common_idx)] + [
                (s, common_idx[reg.reindex(common_idx) == s])
                for s in ["WH", "SH", "SX"]]:
            if len(idx) < 5:
                continue
            ps = []
            for me in eig.columns:
                r, p = spearmanr(contrib.loc[idx], eig.loc[idx, me])
                rows.append(dict(dataset=ds, scope=scope, module=me,
                                 n=len(idx), rho=r, p=p))
                ps.append(p)
            from statsmodels.stats.multitest import multipletests
            _, fdr, _, _ = multipletests(ps, method="fdr_bh")
            for i, me in enumerate(eig.columns):
                rows[-(len(eig.columns)) + i]["FDR"] = fdr[i]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES, "tab_sec16_feast_module.tsv"),
              sep="\t", index=False)
    sig = df[(df["FDR"] < 0.05) | (df["p"] < 0.01)]
    print('\u603b\u884c\u6570:', len(df), ' \u663e\u8457/\u63d0\u793a\u884c:', len(sig))
    print(sig.round(4).to_string(index=False))
    print('\npooled \u5168\u8868\uff08p<0.1\uff09:')
    print(df[(df["scope"] == "pooled") & (df["p"] < 0.1)].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
