# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact

warnings.simplefilter("ignore")

from analysis_configuration import TABLES
from analyze_fruit_metabolome_modules_and_source_tracking import module_eigengenes
from analyze_fruit_metabolome_microbiome_associations import microbe_matrix, sample_map
import analysis_helpers

NAMES = ["turquoise", "blue", "brown", "yellow", "green", "red",
         "black", "pink", "magenta", "purple"]


def functional_group(fn):
    "curated function \u6587\u672c -> \u529f\u80fd\u7ec4\uff08\u542b'\u975e\u81f4\u75c5'\u4e0d\u7b97\u81f4\u75c5\uff09\u3002"
    if ('\u751f\u9632' in fn) or ('\u4fc3\u751f' in fn) or ("PGPR" in fn):
        return '\u751f\u9632/\u4fc3\u751f'
    if '\u964d\u89e3' in fn:
        return '\u964d\u89e3\u8f6c\u5316'
    if ('\u56fa\u6c2e' in fn) or ('\u785d\u5316' in fn) or ('\u53cd\u785d\u5316' in fn):
        return '\u6c2e\u5faa\u73af'
    if '\u81f4\u75c5' in fn and '\u975e\u81f4\u75c5' not in fn:
        return '\u6f5c\u5728\u81f4\u75c5'
    if "C1" in fn:
        return 'C1/\u7532\u57fa\u8425\u517b'
    if '\u6297\u751f\u7d20' in fn:
        return '\u6297\u751f\u7d20\u4ea7\u751f'
    return '\u5176\u4ed6'


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def genus_module_assoc(eig, fmap):
    inv = {v: k for k, v in fmap.items()}
    pref2reg = {"WH": "WH", "BZ": "SH", "SX": "SX"}
    reg = pd.Series({inv[s]: pref2reg[s.split("_")[0]] for s in inv})
    rows = []
    for ds in ["16S", "ITS"]:
        dat = analysis_helpers.load_dataset(ds)
        X = microbe_matrix(dat, list(inv.keys()), qmp=True)
        X.index = [inv[s] for s in X.index]  # Analysis note: see README and original laboratory scripts for methodological details.
        X = np.log10(X + 1).loc[eig.index]
        for scope in ["pooled"] + reg.unique().tolist():
            idx = X.index if scope == "pooled" \
                else X.index[reg.reindex(X.index) == scope]
            ps = []
            start = len(rows)
            for g in X.columns:
                for me in eig.columns:
                    r, p = spearmanr(X.loc[idx, g], eig.loc[idx, me])
                    rows.append(dict(dataset=ds, scope=scope, genus=g,
                                     module=me, n=len(idx), rho=r, p=p))
                    ps.append(p)
            from statsmodels.stats.multitest import multipletests
            _, fdr, _, _ = multipletests(ps, method="fdr_bh")
            for i in range(len(ps)):
                rows[start + i]["FDR"] = fdr[i]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def module_chem(meta_dir):
    from analyze_fruit_metabolome_microbiome_associations import META_DIR
    cc = pd.read_csv(os.path.join(META_DIR, "chemical_class.csv"),
                     encoding="gbk")
    assign = pd.read_csv(os.path.join(
        META_DIR, "analysis_output", '07_\u76f8\u5173\u6027\u7f51\u7edc\u4e0eWGCNA',
        "WGCNA_module_assignment.csv"))
    a = assign.merge(cc, on="Metabolite", how="left")
    a = a[a["Class"].notna()]
    mods = [m for m in NAMES if m in a["Module"].unique()]
    classes = a["Class"].value_counts().index.tolist()
    ct = pd.crosstab(a["Module"], a["Class"]).reindex(
        index=mods, columns=classes).fillna(0)
    total = int(ct.values.sum())
    rows = []
    for m in mods:
        nm = int(ct.loc[m].sum())
        for c in classes:
            k = int(ct.loc[m, c])
            if k == 0:
                continue
            K = int(ct[c].sum())
            _, p = fisher_exact([[k, nm - k], [K - k, total - K - nm + k]])
            rows.append(dict(module=m, chem_class=c, n_in_module=k,
                             n_module=nm,
                             enrich=(k / nm) / (K / total), p=p))
    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(df["p"], method="fdr_bh")
    df["FDR"] = fdr
    return df.sort_values(["module", "p"])


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def fruit_role(assoc):
    pooled = assoc[(assoc.scope == "pooled")].copy()
    out = []
    for ds in ["16S", "ITS"]:
        k = pd.read_csv(os.path.join(TABLES,
                                     "tab_sec15_keystone_%s.tsv" % ds),
                        sep="\t")
        f = k[k.scope == "F"].copy()
        f = f[~f["function"].astype(str).str.contains('\u672a\u6ce8\u91ca')]
        f["func_group"] = f["function"].map(functional_group)
        sub = pooled[pooled.dataset == ds].merge(
            f[["genus", "strategy", "keystone", "function",
               "func_group", "indval_compartment"]],
            on="genus", how="inner")
        # Analysis note: see README and original laboratory scripts for methodological details.
        sub["ar"] = sub["rho"].abs()
        top = sub.loc[sub.groupby("genus")["ar"].idxmax()].copy()
        top["dataset"] = ds
        out.append(top)
    role = pd.concat(out, ignore_index=True)
    role = role[["dataset", "genus", "func_group", "strategy", "keystone",
                 "indval_compartment", "module", "rho", "p", "FDR",
                 "function"]]
    return role.sort_values("FDR").reset_index(drop=True)


def main():
    eig, met, fmap = module_eigengenes()
    assoc = genus_module_assoc(eig, fmap)
    assoc.to_csv(os.path.join(TABLES, "tab_sec16_fruit_genus_module.tsv"),
                 sep="\t", index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    pooled = assoc[assoc.scope == "pooled"]
    mat = pooled.pivot_table(index=["dataset", "genus"], columns="module",
                             values="rho")
    mat = mat[[m for m in ["ME" + n for n in NAMES] if m in mat.columns]]
    fdr_mat = pooled.pivot_table(index=["dataset", "genus"],
                                 columns="module", values="FDR")
    mat.to_csv(os.path.join(
        TABLES, "tab_sec16_fruit_genus_module_heatmap.tsv"), sep="\t")

    from analyze_fruit_metabolome_microbiome_associations import META_DIR
    chem = module_chem(META_DIR)
    chem.to_csv(os.path.join(TABLES, "tab_sec16_module_chem.tsv"),
                sep="\t", index=False)

    role = fruit_role(assoc)
    role.to_csv(os.path.join(TABLES, "tab_sec16_fruit_role.tsv"),
                sep="\t", index=False)

    sig = assoc[(assoc.scope == "pooled") & (assoc.FDR < 0.05)]
    print('\u5173\u8054\u603b\u884c\u6570:', len(assoc), " pooled FDR<0.05:", len(sig))
    print('\n== pooled \u663e\u8457\u5173\u8054 top20\uff08\u6309|rho|\uff09 ==')
    print(sig.reindex(sig.rho.abs().sort_values(ascending=False).index)
          .head(20).round(4).to_string(index=False))
    print('\n== \u5206\u6837\u70b9 FDR<0.05 \u8ba1\u6570 ==')
    print(assoc[assoc.FDR < 0.05].groupby(["dataset", "scope"]).size()
          .to_string())
    print('\n== \u6a21\u5757\u5316\u5b66\u5bcc\u96c6\uff08FDR<0.05\uff09 ==')
    print(chem[chem.FDR < 0.05].round(3).to_string(index=False))
    print('\n== \u679c\u5b9e\u89d2\u8272\u8868\uff08\u6709\u529f\u80fd\u6ce8\u91ca\u5c5e x \u6700\u5f3a\u6a21\u5757\uff09 ==')
    print(role.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
