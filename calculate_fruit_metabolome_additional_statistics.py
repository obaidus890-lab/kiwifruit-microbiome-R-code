# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import io
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

warnings.simplefilter("ignore")
from analysis_configuration import TABLES
import analysis_helpers
from analyze_fruit_metabolome_microbiome_associations import (sample_map, microbe_matrix, mantel,
                               partial_mantel, bc_dist, euclid_dist)


def permanova_dist(D, grp, nperm=999, seed=42):
    '\u5355\u56e0\u5b50 PERMANOVA\uff08adonis2 \u7b49\u4ef7\uff09\uff1a\u8fd4\u56de (R2, F, p)\u3002'
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    grp = np.asarray(grp)
    levels = sorted(set(grp))
    iu = np.triu_indices(N, 1)
    d = D[iu]

    def ss_within(g):
        s = 0.0
        for lv in levels:
            idx = np.where(g == lv)[0]
            m = len(idx)
            if m > 1:
                dd = D[np.ix_(idx, idx)][np.triu_indices(m, 1)]
                s += (dd ** 2).sum() / m
        return s
    ss_total = (d ** 2).sum() / N
    ss_w = ss_within(grp)
    ss_a = ss_total - ss_w
    df1 = len(levels) - 1
    df2 = N - len(levels)
    F = (ss_a / df1) / (ss_w / df2)
    R2 = ss_a / ss_total
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(nperm):
        gp = rng.permutation(grp)
        sw = ss_within(gp)
        sa = ss_total - sw
        Fp = (sa / df1) / (sw / df2)
        if Fp >= F:
            cnt += 1
    return R2, F, (cnt + 1) / (nperm + 1)


def pcoa(D):
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    J = np.eye(N) - 1.0 / N
    G = -0.5 * J.dot(D ** 2).dot(J)
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    keep = w > 1e-9
    return V[:, keep] * np.sqrt(w[keep])[None, :]


def betadisper(D, grp, nperm=999, seed=42):
    '\u7ec4\u5185\u5230\u8d28\u5fc3\u8ddd\u79bb\u7684\u79bb\u6563\u5ea6\u9f50\u6027\u7f6e\u6362\u68c0\u9a8c\uff08vegan betadisper+permutest \u7b49\u4ef7\uff09\u3002'
    coords = pcoa(D)
    grp = np.asarray(grp)
    levels = sorted(set(grp))

    def cdist(g):
        dd = np.zeros(len(g))
        for lv in levels:
            m = g == lv
            cent = coords[m].mean(0)
            dd[m] = np.sqrt(((coords[m] - cent) ** 2).sum(1))
        return dd

    def Fstat(dd, g):
        gm = dd.mean()
        sa = sum((dd[g == lv].mean() - gm) ** 2 * (g == lv).sum() for lv in levels)
        sw = sum(((dd[g == lv] - dd[g == lv].mean()) ** 2).sum() for lv in levels)
        df1 = len(levels) - 1
        df2 = len(g) - len(levels)
        return (sa / df1) / (sw / df2) if sw > 1e-12 and df2 > 0 else np.nan
    dd = cdist(grp)
    fobs = Fstat(dd, grp)
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(nperm):
        gp = rng.permutation(grp)
        dd2 = cdist(gp)
        if Fstat(dd2, gp) >= fobs:
            cnt += 1
    mean_disp = {lv: float(dd[grp == lv].mean()) for lv in levels}
    return fobs, (cnt + 1) / (nperm + 1), mean_disp


def main():
    met, groups, fmap = sample_map()
    X = met.T.loc[sorted(fmap.keys())]
    g = pd.Series({s: groups[s] for s in X.index})
    pm = X[X > 0].values.min() / 2
    L = np.log2(X.where(X > 0, pm))
    U = ((L - L.mean()) / L.std(ddof=1)).fillna(0)
    dm_met_pool = euclid_dist(U)

    d16 = analysis_helpers.load_dataset("16S")
    dit = analysis_helpers.load_dataset("ITS")
    samples = [fmap[s] for s in X.index]
    Xb = microbe_matrix(d16, samples, qmp=True)
    Xf = microbe_matrix(dit, samples, qmp=True)
    dm_b = bc_dist(Xb)
    dm_f = bc_dist(Xf)

    rows = []
    # Analysis note: see README and original laboratory scripts for methodological details.
    rb, pb = mantel(dm_met_pool, dm_b, nperm=999)
    rf, pf = mantel(dm_met_pool, dm_f, nperm=999)
    prb, ppb = partial_mantel(dm_met_pool, dm_b, dm_f, nperm=999)
    prf, ppf = partial_mantel(dm_met_pool, dm_f, dm_b, nperm=999)
    rows.append(dict(scope="pooled", kingdom="bacteria", r=rb, p=pb,
                     partial_r=prb, partial_p=ppb))
    rows.append(dict(scope="pooled", kingdom="fungi", r=rf, p=pf,
                     partial_r=prf, partial_p=ppf))
    # Analysis note: see README and original laboratory scripts for methodological details.
    for site in ["WH", "BZ", "SX"]:
        idx = [s for s in X.index if g[s] == site]
        Us = ((L.loc[idx] - L.loc[idx].mean()) /
              L.loc[idx].std(ddof=1)).fillna(0)
        dsm = euclid_dist(Us)
        dsamp = [fmap[s] for s in idx]
        xb = bc_dist(microbe_matrix(d16, dsamp, qmp=True))
        xf = bc_dist(microbe_matrix(dit, dsamp, qmp=True))
        rb, pb = mantel(dsm, xb, nperm=999)
        rf, pf = mantel(dsm, xf, nperm=999)
        prb, ppb = partial_mantel(dsm, xb, xf, nperm=999)
        prf, ppf = partial_mantel(dsm, xf, xb, nperm=999)
        disp = "SH" if site == "BZ" else site
        rows.append(dict(scope=disp, kingdom="bacteria", r=rb, p=pb,
                         partial_r=prb, partial_p=ppb))
        rows.append(dict(scope=disp, kingdom="fungi", r=rf, p=pf,
                         partial_r=prf, partial_p=ppf))
    mf = pd.DataFrame(rows)
    mf.to_csv(os.path.join(TABLES, "tab_sec16_qmp_mantel_full.tsv"),
              sep="\t", index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    # Analysis note: see README and original laboratory scripts for methodological details.
    D_met = squareform(dm_met_pool)
    grpv = np.asarray(["SH" if v == "BZ" else v for v in g.values])
    R2, F, p = permanova_dist(D_met, grpv, nperm=999)
    fb, pbet, disp = betadisper(D_met, grpv, nperm=999)
    pd.DataFrame([dict(item="PERMANOVA orchard", distance="Euclidean(UV log2)",
                       n=27, groups=3, stat="R2=%.4f F=%.3f p=%.3f" % (R2, F, p),
                       note="adonis2-equivalent, 999 perm"),
                  dict(item="betadisper (homogeneity of dispersions)",
                       distance="Euclidean(UV log2)", n=27, groups=3,
                       stat="F=%.3f p=%.3f" % (fb, pbet),
                       note="mean dispersion by orchard: " +
                            "; ".join("%s=%.3f" % (k, v)
                                      for k, v in disp.items()))]) \
        .to_csv(os.path.join(TABLES, "tab_sec16_permanova.tsv"),
                sep="\t", index=False)

    # Analysis note: see README and original laboratory scripts for methodological details.
    print('=== qmp Mantel FULL (pooled+WH+SH+SX, \u542b\u504f) ===')
    print(mf.round(3).to_string(index=False))
    print("\n=== PERMANOVA (metabolome Euclidean~orchard) ===")
    print("R2=%.4f  F=%.3f  p=%.3f" % (R2, F, p))
    print("=== betadisper ===")
    print("F=%.3f  p=%.3f  disp=%s" % (fb, pbet,
          {k: round(v, 3) for k, v in disp.items()}))


if __name__ == "__main__":
    main()
