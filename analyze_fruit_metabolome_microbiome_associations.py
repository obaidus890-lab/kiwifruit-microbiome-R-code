# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

warnings.simplefilter("ignore")

import analysis_helpers
from analysis_configuration import ROOT, TABLES

META_DIR = os.path.abspath(os.environ.get(
    "KIWI_METABOLOMICS_DIR", os.path.join(os.environ.get("KIWI_ROOT", os.path.dirname(__file__)), "metabolomics")))

# Analysis note: see README and original laboratory scripts for methodological details.
MET_TO_REG = {"WH": "WH", "BZ": "SH", "SX": "SX"}


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def load_metabolome():
    met = pd.read_csv(os.path.join(META_DIR, "metabolome.csv"),
                      encoding="gbk").set_index("Metabolite")
    grp = pd.read_csv(os.path.join(META_DIR, "group_info.csv"), header=None,
                      names=["Sample", "Group", "Date"])
    return met, grp.set_index("Sample")["Group"].to_dict()


def load_microbe(ds, samples):
    '\u8fd4\u56de\u6307\u5b9a\u6837\u672c\u7684\u5c5e\u6c34\u5e73\u7edd\u5bf9\u8f7d\u91cf\u8868\uff08taxa x samples\uff09\u3002'
    dat = analysis_helpers.load_dataset(ds)
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")
    meta = dat["meta"]
    G = G.loc[:, samples].copy()
    X = G.T
    prev = (X > 0).mean(axis=0)
    keep = prev[prev >= PREV_F].index
    return G.loc[keep, samples], dat


def sample_map():
    '\u4ee3\u8c22\u7ec4\u6837\u672c\u540d -> \u5fae\u751f\u7269\u7ec4 F \u6837\u672c\u540d\uff08\u6309\u7ec4\u5185 replicate \u5bf9\u9f50\uff09\u3002'
    met, groups = load_metabolome()
    d16 = analysis_helpers.load_dataset("16S")
    m = d16["meta"]
    fmap = {}
    for s in met.columns:
        reg = MET_TO_REG[groups[s]]
        rep = int(re.sub(r'\D', '', s))
        # Analysis note: see README and original laboratory scripts for methodological details.
        pref = {"WH": "WH", "SH": "BZ", "SX": "SX"}[reg]
        sid = "%s_F_%d" % (pref, rep)
        assert sid in m.index, sid
        fmap[s] = sid
    return met, groups, fmap


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def bc_dist(X):
    '\u6837\u672c x \u7279\u5f81\u77e9\u9635 -> Bray-Curtis \u8ddd\u79bb\u5411\u91cf\u3002'
    return pdist(X.values.astype(float), metric="braycurtis")


def euclid_dist(X):
    return pdist(X.values.astype(float), metric="euclidean")


def pcoa(D):
    '\u7ecf\u5178 PCoA\uff0c\u8fd4\u56de\u4e3b\u5750\u6807\uff08\u6309\u7279\u5f81\u503c\u964d\u5e8f\uff09\u3002'
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D ** 2).dot(H)
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    w = np.clip(w, 0, None)
    return v * np.sqrt(w)[None, :]


def pca_scores(X, k=PCA_AXES):
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Z = np.nan_to_num(Z)
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    return u[:, :k] * s[:k][None, :]


# ------------------------------------------------------------------ #
# Procrustes / Mantel
# ------------------------------------------------------------------ #
def procrustes(A, B, k=None, nperm=NPERM):
    '\u5bf9\u79f0 Procrustes\uff08\u6807\u51c6\u5316\u5230\u5355\u4f4d\u5c3a\u5ea6\uff09\uff0c\u8fd4\u56de r, M2, p\u3002'
    A, B = np.asarray(A, float), np.asarray(B, float)
    if k is not None:
        A, B = A[:, :k], B[:, :k]
    # Analysis note: see README and original laboratory scripts for methodological details.
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)

    def m2(A, B):
        u, _, vt = np.linalg.svd(A.T.dot(B))
        R = u.dot(vt)
        return float(((B - A.dot(R)) ** 2).sum())

    m2_obs = m2(A, B)
    r = np.sqrt(max(0.0, 1 - m2_obs))
    cnt = 0
    for _ in range(nperm):
        m2_r = m2(A, B[RNG.permutation(len(B))])
        if m2_r <= m2_obs:
            cnt += 1
    return r, m2_obs, (cnt + 1) / (nperm + 1)


def mantel(dm_x, dm_y, nperm=NPERM):
    'dm \u4e3a\u538b\u7f29\u8ddd\u79bb\u5411\u91cf\uff1bSpearman Mantel\u3002'
    r_obs = spearmanr(dm_x, dm_y)[0]
    cnt = 0
    sq_x = squareform(dm_x)
    n = sq_x.shape[0]
    tri = np.triu_indices(n, 1)
    for _ in range(nperm):
        p = RNG.permutation(n)
        mp = sq_x[np.ix_(p, p)]
        r = spearmanr(mp[tri], dm_y)[0]
        if r >= r_obs:
            cnt += 1
    return r_obs, (cnt + 1) / (nperm + 1)


def partial_mantel(dm_y, dm_x1, dm_x2, nperm=NPERM):
    """Research analysis helper. See README for pipeline and methodology."""
    def partial_r(ry1, ry2, r12):
        den = np.sqrt(max(1e-12, (1 - ry2 ** 2) * (1 - r12 ** 2)))
        return (ry1 - ry2 * r12) / den

    n = len(squareform(dm_y))
    sq_y, sq_1, sq_2 = map(squareform, (dm_y, dm_x1, dm_x2))
    tri = np.triu_indices(n, 1)

    def stats(p=None):
        if p is not None:
            sq1p = sq_1[np.ix_(p, p)]
        else:
            sq1p = sq_1
        ry1 = spearmanr(sq_y[tri], sq1p[tri])[0]
        ry2 = spearmanr(sq_y[tri], sq_2[tri])[0]
        r12 = spearmanr(sq1p[tri], sq_2[tri])[0]
        return partial_r(ry1, ry2, r12)

    r_obs = stats()
    cnt = 0
    for _ in range(nperm):
        if stats(RNG.permutation(n)) >= r_obs:
            cnt += 1
    return r_obs, (cnt + 1) / (nperm + 1)


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def rda_adjusted_r2(y, x, nperm=199):
    '\u591a\u5143\u56de\u5f52 R2 \u53ca\u7f6e\u6362\u8c03\u6574 R2\uff08p \u503c\uff09\u3002'
    from sklearn.linear_model import LinearRegression
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    reg = LinearRegression().fit(x, y)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
    r2 = 1 - ((y - reg.predict(x)) ** 2).sum() / ss_tot
    n, p = x.shape
    adj = 1 - (1 - r2) * (n - 1) / max(1, n - p - 1)
    cnt = 0
    for _ in range(nperm):
        xp = x[RNG.permutation(n)]
        rp = LinearRegression().fit(xp, y)
        r2p = 1 - ((y - rp.predict(xp)) ** 2).sum() / ss_tot
        if r2p >= r2:
            cnt += 1
    return r2, adj, (cnt + 1) / (nperm + 1)


def forward_rda(y, cands, max_k=8, nperm=199):
    '\u524d\u5411\u9009\u62e9\u89e3\u91ca\u53d8\u91cf\uff0c\u6309\u8c03\u6574 R2 \u589e\u91cf\u3002\u8fd4\u56de\u9009\u4e2d\u5217\u8868\u4e0e\u7d2f\u8ba1\u89e3\u91ca\u7387\u3002'
    remain = list(cands.columns)
    chosen = []
    base_adj = 0.0
    for _ in range(min(max_k, len(remain))):
        best = None
        for c in remain:
            x = cands[chosen + [c]].values if chosen \
                else cands[[c]].values
            _, adj, p = rda_adjusted_r2(y, x, nperm=nperm)
            if best is None or adj > best[1]:
                best = (c, adj, p)
        c, adj, p = best
        if adj <= base_adj + 0.001 or p > 0.05:
            break
        chosen.append(c)
        remain.remove(c)
        base_adj = adj
    x = cands[chosen].values if chosen else np.zeros((len(y), 1))
    r2, adj, p = rda_adjusted_r2(y, x, nperm=nperm) if chosen \
        else (0.0, 0.0, 1.0)
    return chosen, r2, adj, p


# ------------------------------------------------------------------ #
# Analysis note: see README and original laboratory scripts for methodological details.
# ------------------------------------------------------------------ #
def microbe_matrix(dat, samples, qmp=True):
    '\u5c5e\u6c34\u5e73\u77e9\u9635\uff08\u6837\u672c x \u5c5e\uff09\uff1aQMP=copies/g\uff0cRMP=\u884c\u5f52\u4e00\u5316\u3002'
    G = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "genus")
    X = G.loc[:, samples].T
    prev = (X > 0).mean(axis=0)
    X = X[prev[prev >= PREV_F].index]
    if not qmp:
        X = X.div(X.sum(axis=1), axis=0)
    return X


def phylum_loads(dat, samples, qmp=True, top=TOP_PHYLA):
    P = analysis_helpers.agg_tax(dat["asv_abs"], dat["tax"], "phylum")
    P = P.loc[:, samples].T
    order = P.mean().sort_values(ascending=False).index[:top]
    P = P[order]
    if not qmp:
        P = P.div(P.sum(axis=1), axis=0)
    return P


def shannon(X):
    '\u6837\u672c x \u7269\u79cd\u77e9\u9635 -> Shannon\uff08\u57fa\u4e8e\u884c\u5185\u6bd4\u4f8b\uff09\u3002'
    pr = X.div(X.sum(axis=1), axis=0).values
    pr = np.where(pr > 0, pr, 1)
    return -(pr * np.log(pr)).sum(axis=1)


def total_load(dat, samples):
    return dat["asv_abs"][samples].sum(axis=0).values


def run_scope(name, met_mat, m16, mits, fmap, rows_proc, rows_mant):
    '\u5bf9\u67d0\u4e00\u8303\u56f4\uff08pooled \u6216\u5355\u6837\u70b9\uff09\u6267\u884c Procrustes + Mantel\u3002'
    met_s = met_mat.loc[[s for s in met_mat.index]]  # Analysis note: see README and original laboratory scripts for methodological details.
    samples = [fmap[s] for s in met_s.index]
    dm_met = euclid_dist(pd.DataFrame(
        (met_s - met_s.mean()) / met_s.std(ddof=1), index=met_s.index))

    for ds, dat, m in [("bacteria", m16, None), ("fungi", mits, None)]:
        for qmp, tag in [(True, "QMP"), (False, "RMP")]:
            X = microbe_matrix(dat, samples, qmp=qmp)
            dm_b = bc_dist(X)
            # Analysis note: see README and original laboratory scripts for methodological details.
            coords = pcoa(squareform(dm_b))
            metpca = pca_scores(met_s.values)
            r, m2, p = procrustes(coords, metpca, k=PCA_AXES)
            rows_proc.append(dict(scope=name, kingdom=ds, view=tag,
                                  procrustes_r=r, M2=m2, p=p))
            # Mantel
            r_m, p_m = mantel(dm_met, dm_b)
            rows_mant.append(dict(scope=name, kingdom=ds, view=tag,
                                  test="mantel", r=r_m, p=p_m))

    # Analysis note: see README and original laboratory scripts for methodological details.
    for qmp, tag in [(True, "QMP"), (False, "RMP")]:
        Xb = microbe_matrix(m16, samples, qmp=qmp)
        Xf = microbe_matrix(mits, samples, qmp=qmp)
        dm_b, dm_f = bc_dist(Xb), bc_dist(Xf)
        rb, _ = mantel(dm_met, dm_b, nperm=0)
        for k1, dm1, dm2 in [("bacteria", dm_b, dm_f),
                             ("fungi", dm_f, dm_b)]:
            r, p = partial_mantel(dm_met, dm1, dm2)
            rows_mant.append(dict(scope=name, kingdom=k1, view=tag,
                                  test="partial_mantel", r=r, p=p))


def main():
    met, groups, fmap = sample_map()
    met = met.T
    reg = pd.Series({s: MET_TO_REG[groups[s]] for s in met.index})

    m16 = analysis_helpers.load_dataset("16S")
    mits = analysis_helpers.load_dataset("ITS")

    rows_proc, rows_mant, rows_rda = [], [], []

    # ---------- pooled（n=27） ----------
    run_scope("pooled", met, m16, mits, fmap, rows_proc, rows_mant)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for site in ["WH", "SH", "SX"]:
        sub = met.loc[reg[reg == site].index]
        run_scope(site, sub, m16, mits, fmap, rows_proc, rows_mant)

    # ---------- RDA（pooled, QMP vs RMP） ----------
    samples = [fmap[s] for s in met.index]
    y = pca_scores(met.values, k=PCA_AXES)
    for qmp, tag in [(True, "QMP"), (False, "RMP")]:
        for ds, dat in [("bacteria", m16), ("fungi", mits)]:
            cands = {}
            cands["total_copies"] = np.log10(total_load(dat, samples) + 1)
            X = microbe_matrix(dat, samples, qmp=qmp)
            cands["shannon"] = shannon(X)
            P = phylum_loads(dat, samples, qmp=qmp)
            for c in P.columns:
                cands["phylum_" + str(c)] = np.log10(P[c].values + 1) \
                    if qmp else P[c].values
            cand_df = pd.DataFrame(cands, index=met.index)
            chosen, r2, adj, p = forward_rda(y, cand_df)
            rows_rda.append(dict(scope="pooled", kingdom=ds, view=tag,
                                 n_terms=len(chosen), R2=r2, adj_R2=adj,
                                 p=p, terms=";".join(chosen)))

    pd.DataFrame(rows_proc).to_csv(
        os.path.join(TABLES, "tab_sec16_procrustes.tsv"),
        sep="\t", index=False)
    pd.DataFrame(rows_mant).to_csv(
        os.path.join(TABLES, "tab_sec16_mantel.tsv"),
        sep="\t", index=False)
    pd.DataFrame(rows_rda).to_csv(
        os.path.join(TABLES, "tab_sec16_rda.tsv"),
        sep="\t", index=False)

    for tag, df in [("procrustes", rows_proc), ("mantel", rows_mant),
                    ("rda", rows_rda)]:
        print("==== %s ====" % tag)
        print(pd.DataFrame(df).round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
