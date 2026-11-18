# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes as scipy_procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from scipy.stats import t as tdist

import analysis_configuration as C
import analysis_helpers as U

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Analysis note: see README and original laboratory scripts for methodological details.

# Analysis note: see README and original laboratory scripts for methodological details.
COL_ABS = "#E41A1C"      # Analysis note: see README and original laboratory scripts for methodological details.
COL_REL = "#377EB8"      # Analysis note: see README and original laboratory scripts for methodological details.
COL_CI  = "#CCCCCC"      # Analysis note: see README and original laboratory scripts for methodological details.
# Analysis note: see README and original laboratory scripts for methodological details.
TAXA_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488",
               "#F39B7F", "#8491B4", "#91D1C2", "#7E6148"]
# Analysis note: see README and original laboratory scripts for methodological details.
COMP_COLORS = {"BS": "#A0C8E0", "RS": "#1878B0", "R": "#B0D888",
               "S": "#30A028", "L": "#F89898", "F": "#E01818"}

# Analysis note: see README and original laboratory scripts for methodological details.
TOP_N_PHYLA = 6
TOP_N_GENUS = 8
CORR_TOP_N = 25
PGPR_GENERA = {
    "Rhizobium": "N-fix", "Bradyrhizobium": "N-fix",
    "Azospirillum": "N-fix", "Frankia": "N-fix",
    "Pseudomonas": "PGPR", "Bacillus": "PGPR",
    "Burkholderia": "PGPR", "Enterobacter": "PGPR",
    "Trichoderma": "biocontrol", "Penicillium": "decomposer",
    "Fusarium": "pathogen", "Alternaria": "pathogen",
    "Botryosphaeria": "pathogen", "Colletotrichum": "pathogen",
}
RNG = np.random.default_rng(42)


# Analysis note: see README and original laboratory scripts for methodological details.
def load_meta():
    meta = pd.read_csv(os.path.join(C.R_IN, "sample_meta.tsv"), sep="\t")
    meta["region"] = meta["region"].replace(C.REG_MAP)
    return meta.set_index("sample_id")


def load_all(meta):
    out = {}
    for ds in ("16S", "ITS"):
        d = {}
        for lvl in ("phylum", "genus"):
            for kind in ("abs", "rel"):
                p = os.path.join(C.DATA, "%s_%s_%s.tsv" % (lvl, kind, ds))
                df = pd.read_csv(p, sep="\t").set_index("sample_id").astype(float)
                d["%s_%s" % (lvl, kind)] = df
        out[ds] = d
    return out


# ============================================================ helpers ====
def med_comp(df, comp_series):
    '\u6309 compartment \u53d6\u4e2d\u4f4d\u6570\uff0c\u8fd4\u56de 6\xd7taxa\u3002'
    return df.groupby(comp_series).median().loc[C.COMP_ORDER]


def mean_sem(df, comp_series):
    '\u6309 compartment \u53d6\u5747\u503c\u548c SEM\uff0c\u8fd4\u56de (mean 6\xd7taxa, sem 6\xd7taxa)\u3002'
    grp = df.groupby(comp_series)
    m = grp.mean().loc[C.COMP_ORDER]
    s = grp.std(ddof=1).loc[C.COMP_ORDER]
    n = grp.size().loc[C.COMP_ORDER]
    sem = s.div(np.sqrt(n.replace(0, 1)), axis=0)
    return m, sem


def median_norm(v):
    v = np.asarray(v, dtype=float)
    m = np.median(v)
    if m > 0:
        return v / m
    mx = v.max()
    return v / mx if mx > 0 else v


def hellinger(df):
    rs = df.sum(axis=1)
    rel = df.div(rs.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


def pcoa(dm):
    n = dm.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dm ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = eigvals > 0
    ev, vec = eigvals[pos][:2], eigvecs[:, pos][:, :2]
    coords = vec * np.sqrt(ev)
    exp_ = ev / eigvals[eigvals > 0].sum() * 100
    return coords, exp_


def ph(name):
    alias = {"Pseudomonadota": "Proteobacteria",
             "Actinomycetota": "Actinobacteria",
             "Acidobacteriota": "Acidobacteria",
             "Bacteroidota": "Bacteroidetes",
             "Bacillota": "Firmicutes"}
    return alias.get(name, name)


def panel_letter(ax, letter, x=-0.18, y=1.08, fontsize=11, upper=True):
    """Add a lowercase label to the upper left of a plot panel."""
    ax.text(x, y, "%s)" % str(letter).lower(),
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top",
            family="serif")


# Analysis note: see README and original laboratory scripts for methodological details.
def identify_dilution(med_abs, med_rel):
    rows = []
    x = np.arange(len(C.COMP_ORDER), dtype=float)
    for taxon in med_abs.columns:
        a = med_abs[taxon].values.astype(float)
        r = med_rel[taxon].values.astype(float)
        na, nr = median_norm(a), median_norm(r)
        sa = np.polyfit(x, na, 1)[0]
        sr = np.polyfit(x, nr, 1)[0]
        score = sr - sa
        rows.append({"taxon": taxon, "abs_slope": sa, "rel_slope": sr,
                      "dilution_score": score,
                      "mean_abs": float(med_abs[taxon].mean()),
                      "mean_rel": float(med_rel[taxon].mean())})
    return pd.DataFrame(rows).sort_values("dilution_score", ascending=False)


def max_norm(v):
    'max-normalize\uff1a\u5c06\u8f68\u8ff9\u7f29\u653e\u81f3 [0, 1]\uff0c\u9002\u5408\u8de8\u7c7b\u7fa4\u6bd4\u8f83\u3002'
    v = np.asarray(v, dtype=float)
    mx = v.max()
    return v / mx if mx > 0 else v


def select_phyla(phy_abs, phy_rel, comp_series, n=TOP_N_PHYLA):
    med_r = med_comp(phy_rel, comp_series)
    med_a = med_comp(phy_abs, comp_series)
    overall = med_r.mean(axis=0).sort_values(ascending=False)
    sel = list(overall.index[:n])
    # Analysis note: see README and original laboratory scripts for methodological details.
    div = {}
    for g in overall.index:
        if g in sel:
            continue
        na = max_norm(med_a[g].values)
        nr = max_norm(med_r[g].values)
        div[g] = float(np.mean(np.abs(na - nr)))
    div_s = pd.Series(div).sort_values(ascending=False)
    for g in div_s.index[:2]:
        if g not in sel and len(sel) < n + 2:
            sel.append(g)
    return sel


# Analysis note: see README and original laboratory scripts for methodological details.
MARKERS = ["o", "s", "^", "D", "v"]


def select_top_taxa(df_rel, meta, region, n=5):
    '\u5728\u6307\u5b9a\u5730\u533a\u5185\u6309\u5e73\u5747\u76f8\u5bf9\u4e30\u5ea6\u9009 top n \u7c7b\u7fa4\uff08\u4e0d\u5408\u5e76\u3001\u4e0d\u8865 Other\uff09\u3002'
    mask = meta["region"] == region
    sub = df_rel.loc[mask]
    comp = meta.loc[mask, "compartment"]
    g = sub.groupby(comp).mean().reindex(C.COMP_ORDER).fillna(0.0)
    return list(g.mean(axis=0).sort_values(ascending=False).index[:n])


def plot_region_trajectory(ax, df_abs, df_rel, meta, region, taxa, title,
                           show_xlabel=True, show_ylabel=True):
    """Research analysis helper. See README for pipeline and methodology."""
    mask = meta["region"] == region
    sub_a, sub_r = df_abs.loc[mask], df_rel.loc[mask]
    comp = meta.loc[mask, "compartment"]
    ma = sub_a.groupby(comp).median().reindex(C.COMP_ORDER).fillna(0.0)
    mr = sub_r.groupby(comp).median().reindex(C.COMP_ORDER).fillna(0.0)
    x = np.arange(len(C.COMP_ORDER))

    for i, taxon in enumerate(taxa):
        mk = MARKERS[i % len(MARKERS)]
        na = max_norm(ma[taxon].values)
        nr = max_norm(mr[taxon].values)
        ax.plot(x, na, "-", color=COL_ABS, lw=1.5, zorder=3)
        ax.plot(x, na, marker=mk, color=COL_ABS, mfc=COL_ABS, mec=COL_ABS,
                ms=4.5, lw=0, zorder=4)
        ax.plot(x, nr, "--", color=COL_REL, lw=1.1, zorder=2)
        ax.plot(x, nr, marker=mk, color=COL_REL, mfc="white", mec=COL_REL,
                ms=4.0, mew=1.1, lw=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(C.COMP_LABELS, rotation=40, ha="right", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Normalized abundance (0-1)", fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Compartment", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.3, len(C.COMP_ORDER) - 0.7)
    ax.set_ylim(-0.05, 1.12)

    handles = [Line2D([0], [0], marker=MARKERS[i % len(MARKERS)], mfc="0.4",
                      mec="0.4", ms=4.5, lw=0, label=ph(t))
               for i, t in enumerate(taxa)]
    handles += [Line2D([0], [0], color=COL_ABS, lw=1.5, label="AA (absolute)"),
                Line2D([0], [0], color=COL_REL, lw=1.1, ls="--",
                        label="RA (relative)")]
    ax.legend(handles=handles, fontsize=5.5, frameon=True,
              edgecolor="#CCCCCC", fancybox=False, loc="upper right",
              handlelength=1.6, labelspacing=0.25, borderpad=0.3, ncol=1)


# ========================================================= §2.2 Procrustes ====
def procrustes_test(co_a, co_r, n_perm=999):
    """Research analysis helper. See README for pipeline and methodology."""
    _, _, m2_obs = scipy_procrustes(co_a, co_r)
    m2_obs = float(m2_obs)
    r_obs = np.sqrt(max(1.0 - m2_obs, 0.0))
    cnt = 0
    n = co_r.shape[0]
    for _ in range(n_perm):
        perm = RNG.permutation(n)
        _, _, m2_p = scipy_procrustes(co_a, co_r[perm])
        if float(m2_p) <= m2_obs:
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)
    return m2_obs, r_obs, p_perm


def run_procrustes_all(D, meta):
    rows, coords_out = [], {}
    for ds in ("16S", "ITS"):
        d = D[ds]
        pa, pr_ = d["phylum_abs"], d["phylum_rel"]
        common = pa.index.intersection(pr_.index)
        pa, pr_ = pa.loc[common], pr_.loc[common]

        h_abs, h_rel = hellinger(pa), hellinger(pr_)
        dm_a = squareform(pdist(h_abs.values, "braycurtis"))
        dm_r = squareform(pdist(h_rel.values, "braycurtis"))
        co_a, exp_a = pcoa(dm_a)
        co_r, exp_r = pcoa(dm_r)

        m2, r_val, p_val = procrustes_test(co_a, co_r)
        rows.append({"dataset": ds, "scope": "overall", "region": "ALL",
                      "n": len(common), "M2": m2, "r": r_val, "p": p_val,
                      "PCo1_abs": exp_a[0], "PCo2_abs": exp_a[1],
                      "PCo1_rel": exp_r[0], "PCo2_rel": exp_r[1]})
        coords_out[ds] = {"abs": co_a, "rel": co_r, "exp_abs": exp_a,
                           "exp_rel": exp_r, "common": common}

        m = meta.loc[common]
        for reg in C.REGIONS:
            idx = [s for s in m[m["region"] == reg].index if s in common]
            if len(idx) < 6:
                continue
            ha = hellinger(pa.loc[idx])
            hr = hellinger(pr_.loc[idx])
            da = squareform(pdist(ha.values, "braycurtis"))
            dr = squareform(pdist(hr.values, "braycurtis"))
            ca, _ = pcoa(da)
            cr, _ = pcoa(dr)
            m2r, rr, pr_val = procrustes_test(ca, cr, n_perm=499)
            rows.append({"dataset": ds, "scope": "regional", "region": reg,
                          "n": len(idx), "M2": m2r, "r": rr, "p": pr_val,
                          "PCo1_abs": np.nan, "PCo2_abs": np.nan,
                          "PCo1_rel": np.nan, "PCo2_rel": np.nan})
    return rows, coords_out


def _align_rel_to_abs(co_a, co_r):
    """Research analysis helper. See README for pipeline and methodology."""
    mu_a = co_a.mean(axis=0)
    A = co_a - mu_a
    B = co_r - co_r.mean(axis=0)
    R, _ = orthogonal_procrustes(B, A)
    B2 = B @ R
    denom = float(np.trace(B2.T @ B2))
    s = float(np.trace(A.T @ B2)) / denom if denom > 0 else 1.0
    return mu_a + B2 * s


def _conf_ellipse(ax, pts, color):
    '95% \u7f6e\u4fe1\u692d\u5706\uff08chi-square 5.991\uff09\u3002'
    from matplotlib.patches import Ellipse
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    chi = 5.991
    w = 2 * np.sqrt(chi * max(eigvals[0], 0))
    h = 2 * np.sqrt(chi * max(eigvals[1], 0))
    ax.add_patch(Ellipse(mu, w, h, angle=angle, facecolor="none",
                         edgecolor=color, lw=1.1, ls="--", alpha=0.85,
                         zorder=2))


def pcoa_coords_subset(pa, pr, idx):
    '\u5bf9\u6307\u5b9a\u6837\u672c\u5b50\u96c6\u8ba1\u7b97 abs/rel \u7684 PCoA \u5750\u6807\u3002'
    pa_s, pr_s = pa.loc[idx], pr.loc[idx]
    ha, hr = hellinger(pa_s), hellinger(pr_s)
    da = squareform(pdist(ha.values, "braycurtis"))
    dr = squareform(pdist(hr.values, "braycurtis"))
    ca, ea = pcoa(da)
    cr, er = pcoa(dr)
    return {"abs": ca, "rel": cr, "exp_abs": ea, "exp_rel": er}


def plot_pcoa_method(ax, coords_d, ds, letter, title=None, letter_upper=True):
    """Research analysis helper. See README for pipeline and methodology."""
    co_a = coords_d["abs"]
    co_r = _align_rel_to_abs(coords_d["abs"], coords_d["rel"])
    exp_a, exp_r = coords_d["exp_abs"], coords_d["exp_rel"]
    for i in range(len(co_a)):
        ax.plot([co_a[i, 0], co_r[i, 0]], [co_a[i, 1], co_r[i, 1]],
                color="#999999", lw=0.5, alpha=0.35, zorder=0)
    ax.scatter(co_a[:, 0], co_a[:, 1], c=COL_ABS, s=24, alpha=0.8,
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(co_r[:, 0], co_r[:, 1], c=COL_REL, s=24, alpha=0.8,
               edgecolors="white", linewidths=0.4, zorder=3)
    _conf_ellipse(ax, co_a, COL_ABS)
    _conf_ellipse(ax, co_r, COL_REL)
    ax.axhline(0, color="#AAAAAA", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="#AAAAAA", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("PCo1 (abs %.1f%% / rel %.1f%%)" % (exp_a[0], exp_r[0]),
                  fontsize=11)
    ax.set_ylabel("PCo2 (abs %.1f%% / rel %.1f%%)" % (exp_a[1], exp_r[1]),
                  fontsize=11)
    ax.set_title(title if title else C.DATASETS[ds],
                 fontsize=12, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", mfc=COL_ABS, mec=COL_ABS, ms=6,
                      lw=0, label="QMP (AA)"),
               Line2D([0], [0], marker="o", mfc=COL_REL, mec=COL_REL, ms=6,
                      lw=0, label="RMP (RA)")]
    ax.legend(handles=handles, fontsize=9, frameon=True, edgecolor="#CCCCCC",
              fancybox=False, loc="upper left", handlelength=1.4)
    if letter:
        panel_letter(ax, letter, fontsize=16, upper=letter_upper)


# ========================================================= §2.3 Spearman ====
def genus_correlations(D, meta, top_n=CORR_TOP_N):
    out = {}
    for ds in ("16S", "ITS"):
        ga, gr = D[ds]["genus_abs"], D[ds]["genus_rel"]
        common = ga.index.intersection(gr.index)
        ga, gr = ga.loc[common], gr.loc[common]
        top = ga.mean(axis=0).sort_values(ascending=False).index[:top_n]
        ga_t, gr_t = ga[list(top)], gr[list(top)]
        out[ds] = {}
        out[ds]["overall"] = {"abs": ga_t.corr("spearman"),
                               "rel": gr_t.corr("spearman"),
                               "genera": list(top)}
        m = meta.loc[common]
        for reg in C.REGIONS:
            idx = [s for s in m[m["region"] == reg].index if s in common]
            if len(idx) < 6:
                continue
            out[ds][reg] = {"abs": ga_t.loc[idx].corr("spearman"),
                             "rel": gr_t.loc[idx].corr("spearman"),
                             "genera": list(top)}
    return out


def corr_consistency(D, meta, corr_d, top_n=CORR_TOP_N):
    rows = []
    for ds in ("16S", "ITS"):
        ga, gr = D[ds]["genus_abs"], D[ds]["genus_rel"]
        common = ga.index.intersection(gr.index)
        ga, gr = ga.loc[common], gr.loc[common]
        top = corr_d[ds]["overall"]["genera"]
        ga_t, gr_t = ga[list(top)], gr[list(top)]
        n_g = len(top)
        iu = np.triu_indices(n_g, k=1)

        for scope in ["overall"] + C.REGIONS:
            if scope == "overall":
                ga_s, gr_s = ga_t, gr_t
            else:
                m = meta.loc[common]
                idx = [s for s in m[m["region"] == scope].index if s in common]
                if len(idx) < 6:
                    continue
                ga_s, gr_s = ga_t.loc[idx], gr_t.loc[idx]

            n_total = len(iu[0])

            # Analysis note: see README and original laboratory scripts for methodological details.
            ra_arr = np.empty(n_total)
            pa_arr = np.empty(n_total)
            rr_arr = np.empty(n_total)
            pr_arr = np.empty(n_total)
            for k, (ii, jj) in enumerate(zip(iu[0], iu[1])):
                va1, va2 = ga_s.iloc[:, ii].values, ga_s.iloc[:, jj].values
                vr1, vr2 = gr_s.iloc[:, ii].values, gr_s.iloc[:, jj].values
                ra_arr[k], pa_arr[k] = spearmanr(va1, va2)
                rr_arr[k], pr_arr[k] = spearmanr(vr1, vr2)
            qa = U.bh_q(pa_arr)
            qr = U.bh_q(pr_arr)
            sig_a = qa < 0.05
            sig_r = qr < 0.05

            n_pos_aa = int(np.sum(sig_a & (ra_arr > 0)))
            n_neg_aa = int(np.sum(sig_a & (ra_arr < 0)))
            n_ns_aa = int(np.sum(~sig_a))
            n_pos_ra = int(np.sum(sig_r & (rr_arr > 0)))
            n_neg_ra = int(np.sum(sig_r & (rr_arr < 0)))
            n_ns_ra = int(np.sum(~sig_r))
            both = sig_a & sig_r
            # consistent = identical AA/RA class (diagonal of the 3x3
            # confusion matrix, incl. NS-NS agreement) - same metric as
            # the confusion-matrix figure panels (D/E/F)
            n_consistent = int(np.sum(
                (both & ((ra_arr > 0) == (rr_arr > 0))) | (~sig_a & ~sig_r)))
            n_flip = int(np.sum(both & ((ra_arr > 0) != (rr_arr > 0))))

            rows.append({
                "dataset": ds, "scope": scope,
                "n_samples": len(ga_s), "n_pairs": n_total,
                "AA_pos": n_pos_aa, "AA_neg": n_neg_aa, "AA_ns": n_ns_aa,
                "RA_pos": n_pos_ra, "RA_neg": n_neg_ra, "RA_ns": n_ns_ra,
                "consistent": n_consistent, "flipped": n_flip,
                "consistent_ratio": n_consistent / max(n_total, 1),
            })
    return pd.DataFrame(rows)


# Analysis note: see README and original laboratory scripts for methodological details.
def plot_corr_table(ax, cons_df, ds, letter):
    """Research analysis helper. See README for pipeline and methodology."""
    sub = cons_df[cons_df["dataset"] == ds].reset_index(drop=True)
    ax.set_title("%s" % C.DATASETS[ds], fontsize=12, fontweight="bold")
    ax.axis("off")

    scopes = ["overall", "WH", "SH", "SX"]
    scope_labels = ["Overall", "WH", "SH", "SX"]
    # Analysis note: see README and original laboratory scripts for methodological details.
    cell_rows = []
    for sc in scopes:
        r = sub[sub["scope"] == sc]
        if r.empty:
            cell_rows.append(None)
        else:
            cell_rows.append(r.iloc[0])

    # Analysis note: see README and original laboratory scripts for methodological details.
    row_defs = [
        ("Pos (p < 0.05)", "pos"),
        ("Neg (p < 0.05)", "neg"),
        ("NS (p >= 0.05)", "ns"),
        ("Consistent (same class)", "consistent"),
        ("Flipped", "flipped"),
    ]

    # Analysis note: see README and original laboratory scripts for methodological details.
    header_aa_ra = []
    for sl in scope_labels:
        header_aa_ra.extend(["AA", "RA"])
    col_labels = ["Metric"] + header_aa_ra

    table_data = []
    for rlabel, rkey in row_defs:
        row = [rlabel]
        for ci, sc in enumerate(scopes):
            cr = cell_rows[ci]
            if cr is None:
                row.extend(["-", "-"])
                continue
            if rkey == "pos":
                row.extend([str(int(cr["AA_pos"])), str(int(cr["RA_pos"]))])
            elif rkey == "neg":
                row.extend([str(int(cr["AA_neg"])), str(int(cr["RA_neg"]))])
            elif rkey == "ns":
                row.extend([str(int(cr["AA_ns"])), str(int(cr["RA_ns"]))])
            elif rkey == "consistent":
                pct_ = cr["consistent_ratio"] * 100
                val = "%d (%.0f%%)" % (int(cr["consistent"]), pct_)
                row.extend([val, "-"])
            elif rkey == "flipped":
                row.extend([str(int(cr["flipped"])), "-"])
        table_data.append(row)

    # Analysis note: see README and original laboratory scripts for methodological details.
    meta_row = ["n (samples)"]
    pairs_row = ["n (pairs)"]
    ratio_row = ["Pos ratio AA/RA"]
    for ci, sc in enumerate(scopes):
        cr = cell_rows[ci]
        if cr is None:
            meta_row.extend(["-", "-"])
            pairs_row.extend(["-", "-"])
            ratio_row.extend(["-", "-"])
        else:
            ns = str(int(cr["n_samples"]))
            np_ = str(int(cr["n_pairs"]))
            aa_p = int(cr["AA_pos"])
            ra_p = int(cr["RA_pos"])
            ratio = "%.2f" % (aa_p / max(ra_p, 1)) if ra_p > 0 else "-"
            meta_row.extend([ns, ns])
            pairs_row.extend([np_, np_])
            ratio_row.extend([ratio, "-"])
    table_data.extend([meta_row, pairs_row, ratio_row])

    # Analysis note: see README and original laboratory scripts for methodological details.
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.8)

    n_data_rows = len(row_defs)
    n_total_rows = len(table_data) + 1  # +1 for header
    n_cols = len(col_labels)

    for (row_i, col_j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#AAAAAA")
        cell.set_linewidth(0.5)
        if row_i == 0:
            # Analysis note: see README and original laboratory scripts for methodological details.
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
            cell.set_height(0.07)
        elif col_j == 0:
            # Analysis note: see README and original laboratory scripts for methodological details.
            if row_i <= n_data_rows:
                cell.set_facecolor("#ECF0F1")
                cell.set_text_props(fontweight="bold", fontsize=7.5, ha="left")
            else:
                cell.set_facecolor("#F8F9FA")
                cell.set_text_props(fontstyle="italic", fontsize=7, ha="left")
        elif row_i <= n_data_rows:
            # Analysis note: see README and original laboratory scripts for methodological details.
            if row_i % 2 == 0:
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_facecolor("white")
            cell.set_text_props(fontsize=7.5)
            # Analysis note: see README and original laboratory scripts for methodological details.
            if row_i == n_data_rows - 1:  # Consistent
                cell.set_facecolor("#D5F5E3")
                cell.set_text_props(fontweight="bold", fontsize=7.5)
            elif row_i == n_data_rows:  # Flipped
                cell.set_facecolor("#FADBD8")
        else:
            # Analysis note: see README and original laboratory scripts for methodological details.
            cell.set_facecolor("#F8F9FA")
            cell.set_text_props(fontstyle="italic", fontsize=6.5)

    # Analysis note: see README and original laboratory scripts for methodological details.
    scope_note = "Spearman |p| > 0, adj. p < 0.05 significant.  " \
                 "AA = absolute abundance; RA = relative abundance."
    ax.text(0.5, -0.03, scope_note, transform=ax.transAxes, fontsize=6,
            ha="center", va="top", color="#888888", style="italic")

    panel_letter(ax, letter, fontsize=16)


# Analysis note: see README and original laboratory scripts for methodological details.
def select_top_genera_overall(df_rel, meta, n=5):
    '\u5168\u5c40\u6309\u5e73\u5747\u76f8\u5bf9\u4e30\u5ea6\u9009 top n \u5c5e\uff08\u8de8\u5730\u533a\u4e00\u81f4\uff0c\u4fbf\u4e8e\u6bd4\u8f83\uff09\u3002'
    comp = meta.loc[df_rel.index, "compartment"]
    g = df_rel.groupby(comp).mean().reindex(C.COMP_ORDER).fillna(0.0)
    return list(g.mean(axis=0).sort_values(ascending=False).index[:n])


def plot_genus_panel(ax, df_abs, df_rel, meta, region, genus,
                     region_title=None, genus_label=False, show_xlabel=False,
                     show_left_ylabel=False, show_right_ylabel=False):
    """Research analysis helper. See README for pipeline and methodology."""
    mask = meta["region"] == region
    comp = meta.loc[mask, "compartment"]
    ga = df_abs.loc[mask, genus].groupby(comp)
    gr = df_rel.loc[mask, genus].groupby(comp)
    ma = ga.mean().reindex(C.COMP_ORDER)
    sa = ga.sem().reindex(C.COMP_ORDER).fillna(0.0)
    mr = gr.mean().reindex(C.COMP_ORDER)
    sr = gr.sem().reindex(C.COMP_ORDER).fillna(0.0)
    x = np.arange(len(C.COMP_ORDER))
    ci = 1.96

    # Analysis note: see README and original laboratory scripts for methodological details.
    ax2 = ax.twinx()
    la = np.log10(ma.clip(lower=1.0))
    lse = (sa / (ma.clip(lower=1.0) * np.log(10))).fillna(0.0)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for i, c in enumerate(C.COMP_ORDER):
        sel = comp == c
        vr = df_rel.loc[mask, genus][sel].dropna().values
        va = df_abs.loc[mask, genus][sel].dropna().values
        if len(vr) > 0:
            jx = np.linspace(-0.15, 0.15, len(vr)) if len(vr) > 1 else np.zeros(1)
            ax.scatter(i + jx, vr, s=8, alpha=0.4, color=COL_REL,
                       edgecolors="none", zorder=2)
        if len(va) > 0:
            jx = np.linspace(-0.15, 0.15, len(va)) if len(va) > 1 else np.zeros(1)
            ax2.scatter(i + jx, np.log10(np.clip(va, 1.0, None)), s=8,
                        alpha=0.4, color=COL_ABS, edgecolors="none", zorder=2)

    # Analysis note: see README and original laboratory scripts for methodological details.
    ax.plot(x, mr, "-", color=COL_REL, lw=1.8, zorder=3)
    ax.fill_between(x, (mr - ci * sr).clip(lower=0), mr + ci * sr,
                    color="0.6", alpha=0.3, lw=0, zorder=1)
    ax.set_ylim(0, max(1.0, float((mr + ci * sr).max()) * 1.1))
    if show_left_ylabel:
        ax.set_ylabel("Relative abundance", color=COL_REL, fontsize=12)
    ax.tick_params(axis="y", labelcolor=COL_REL, labelsize=9)

    # Analysis note: see README and original laboratory scripts for methodological details.
    ax2.plot(x, la, "-", color=COL_ABS, lw=1.8, zorder=3)
    ax2.fill_between(x, la - ci * lse, la + ci * lse,
                     color="0.6", alpha=0.3, lw=0, zorder=1)
    if show_right_ylabel:
        ax2.set_ylabel("Copies (log10)", color=COL_ABS, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COL_ABS, labelsize=9)

    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels(C.COMP_LABELS, rotation=40, ha="right", fontsize=10)
    else:
        ax.set_xticklabels([])
    if region_title:
        ax.set_title(region_title, fontsize=13, fontweight="bold")
    if genus_label:
        ax.text(-0.35, 0.5, genus, rotation=90, va="center", ha="center",
                fontsize=12, style="italic", fontweight="bold",
                transform=ax.transAxes)
    ax.set_xlim(-0.3, len(C.COMP_ORDER) - 0.7)
    ax.tick_params(axis="x", labelsize=10)


CLASS_IDX = {"Pos": 0, "Neg": 1, "NS": 2}
CLASS_NAMES = ["Pos", "Neg", "NS"]


def confusion_aa_ra(ga, gr, meta, genera, scope):
    '\u5bf9 5 \u4e2a\u5c5e\u8ba1\u7b97 AA\xd7RA 3\xd73 \u4e00\u81f4\u6027\u6df7\u6dc6\u77e9\u9635\uff08BH \u6821\u6b63 p<0.05 \u663e\u8457\uff09\u3002'
    ga_t, gr_t = ga[list(genera)], gr[list(genera)]
    if scope != "overall":
        m = meta.loc[ga.index]
        idx = [s for s in m[m["region"] == scope].index if s in ga.index]
        ga_t, gr_t = ga_t.loc[idx], gr_t.loc[idx]
    iu = np.triu_indices(len(genera), k=1)
    n = len(iu[0])
    ra = np.empty(n); pa = np.empty(n); rr = np.empty(n); pr = np.empty(n)
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        ra[k], pa[k] = spearmanr(ga_t.iloc[:, i], ga_t.iloc[:, j])
        rr[k], pr[k] = spearmanr(gr_t.iloc[:, i], gr_t.iloc[:, j])
    qa, qr = U.bh_q(pa), U.bh_q(pr)

    def cls(r, q):
        if q < 0.05:
            return "Pos" if r > 0 else "Neg"
        return "NS"

    mat = np.zeros((3, 3))
    for k in range(n):
        mat[CLASS_IDX[cls(ra[k], qa[k])], CLASS_IDX[cls(rr[k], qr[k])]] += 1
    return mat, n


def plot_confusion(ax, mat, n, title, show_ylabel=True):
    """Build a three-by-three classification heatmap."""
    pct = mat / max(n, 1) * 100.0
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(3):
        for j in range(3):
            color = "#D43F3A" if i == j else "#3FA7C0"
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, facecolor=color,
                                       edgecolor="white", lw=1.2))
            txt = "%.1f" % pct[i, j]
            ax.text(j + 0.5, 2 - i + 0.5, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if i == j else "black")
    # Analysis note: see README and original laboratory scripts for methodological details.
    for j, nm in enumerate(CLASS_NAMES):
        ax.text(j + 0.5, 3.10, nm, ha="center", va="bottom", fontsize=10)
    for i, nm in enumerate(CLASS_NAMES):
        ax.text(-0.12, 2 - i + 0.5, nm, ha="right", va="center", fontsize=10)
    if show_ylabel:
        ax.text(-0.55, 1.5, "Absolute abundance", rotation=90, va="center",
                ha="center", fontsize=10)
    ax.text(1.5, 3.42, "Relative abundance", ha="center", va="bottom",
            fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
    for sp in ax.spines.values():
        sp.set_visible(False)


# Analysis note: see README and original laboratory scripts for methodological details.
def make_trajectory_plate(df_abs, df_rel, meta, taxa, letter, title,
                          italic_taxa=True, letter_upper=True):
    """Research analysis helper. See README for pipeline and methodology."""
    n_r, n_c = len(C.REGIONS), len(taxa)
    fig = plt.figure(figsize=(18.0, 9.4))
    gs = fig.add_gridspec(n_r, n_c, hspace=0.62, wspace=0.52,
                          left=0.05, right=0.955, top=0.885, bottom=0.115)
    for ri, reg in enumerate(C.REGIONS):
        for ci, g in enumerate(taxa):
            ax = fig.add_subplot(gs[ri, ci])
            plot_genus_panel(ax, df_abs, df_rel, meta, reg, g,
                             show_xlabel=(ri == n_r - 1),
                             show_left_ylabel=(ci == 0),
                             show_right_ylabel=(ci == n_c - 1))
            if ri == 0:
                ax.set_title(g, fontsize=14, fontweight="bold",
                             style="italic" if italic_taxa else "normal")
            if ci == 0:
                ax.text(-0.52, 0.5, reg, rotation=90, va="center", ha="center",
                        fontsize=16, fontweight="bold", transform=ax.transAxes)
    panel_letter(fig.axes[0], letter, x=-0.62, y=1.12, fontsize=20,
                 upper=letter_upper)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.015,
             "Red = absolute abundance (right axis, log10 copies); "
             "blue = relative abundance (left axis); dots = individual samples; "
             "shaded bands = 95% CI.",
             ha="center", fontsize=11, style="italic")
    return fig


# Analysis note: see README and original laboratory scripts for methodological details.
def _cls(r, q):
    if q < 0.05:
        return "Pos" if r > 0 else "Neg"
    return "NS"


def confusion_per_taxon(ga, gr, taxa):
    """Research analysis helper. See README for pipeline and methodology."""
    ga_t, gr_t = ga[list(taxa)], gr[list(taxa)]
    mats = []
    for t in taxa:
        others = [o for o in taxa if o != t]
        m = len(others)
        ra = np.empty(m); pa = np.empty(m); rr = np.empty(m); pr = np.empty(m)
        for k, o in enumerate(others):
            ra[k], pa[k] = spearmanr(ga_t[t].values, ga_t[o].values)
            rr[k], pr[k] = spearmanr(gr_t[t].values, gr_t[o].values)
        ra, pa = np.nan_to_num(ra, nan=0.0), np.nan_to_num(pa, nan=1.0)
        rr, pr = np.nan_to_num(rr, nan=0.0), np.nan_to_num(pr, nan=1.0)
        qa, qr = U.bh_q(pa), U.bh_q(pr)
        mat = np.zeros((3, 3))
        for k in range(m):
            mat[CLASS_IDX[_cls(ra[k], qa[k])],
                CLASS_IDX[_cls(rr[k], qr[k])]] += 1
        mats.append(mat)
    return mats


def confusion_overall(ga, gr, taxa):
    """Research analysis helper. See README for pipeline and methodology."""
    ga_t, gr_t = ga[list(taxa)], gr[list(taxa)]
    iu = np.triu_indices(len(taxa), k=1)
    m = len(iu[0])
    ra = np.empty(m); pa = np.empty(m); rr = np.empty(m); pr = np.empty(m)
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        ra[k], pa[k] = spearmanr(ga_t.iloc[:, i], ga_t.iloc[:, j])
        rr[k], pr[k] = spearmanr(gr_t.iloc[:, i], gr_t.iloc[:, j])
    ra, pa = np.nan_to_num(ra, nan=0.0), np.nan_to_num(pa, nan=1.0)
    rr, pr = np.nan_to_num(rr, nan=0.0), np.nan_to_num(pr, nan=1.0)
    qa, qr = U.bh_q(pa), U.bh_q(pr)
    mat = np.zeros((3, 3))
    for k in range(m):
        mat[CLASS_IDX[_cls(ra[k], qa[k])],
            CLASS_IDX[_cls(rr[k], qr[k])]] += 1
    return mat, m


def plot_confusion_facet(ax, mat, n, name, show_ytick=False, italic=False):
    """Build a three-by-three consistency heatmap."""
    pct = mat / max(n, 1) * 100.0
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(CLASS_NAMES[::-1] if show_ytick else [])
    ax.tick_params(length=0, labelsize=10)
    for i in range(3):
        for j in range(3):
            color = "#D43F3A" if i == j else "#3FA7C0"
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, facecolor=color,
                                       edgecolor="white", lw=1.2))
            ax.text(j + 0.5, 2 - i + 0.5, "%.0f" % pct[i, j], ha="center",
                    va="center", fontsize=10, fontweight="bold",
                    color="white" if i == j else "black")
    ax.text(0.5, 1.05, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            style="italic" if italic else "normal",
            bbox=dict(boxstyle="square,pad=0.28", facecolor="#EBEBEB",
                      edgecolor="#999999", lw=0.8))
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_confusion_plate(ds, ph_taxa, gen_taxa, meta, D, letter):
    """Research analysis helper. See README for pipeline and methodology."""
    mats_ph = confusion_per_taxon(D[ds]["phylum_abs"], D[ds]["phylum_rel"],
                                  ph_taxa)
    mats_gen = confusion_per_taxon(D[ds]["genus_abs"], D[ds]["genus_rel"],
                                   gen_taxa)
    mat_ph_all, n_ph_all = confusion_overall(D[ds]["phylum_abs"],
                                             D[ds]["phylum_rel"], ph_taxa)
    # Analysis note: see README and original laboratory scripts for methodological details.
    ga_ = D[ds]["genus_abs"]
    gr_ = D[ds]["genus_rel"]
    common = ga_.index.intersection(gr_.index)
    gen_top = ga_.loc[common].mean(axis=0).sort_values(
        ascending=False).index[:CORR_TOP_N]
    mat_gen_all, n_gen_all = confusion_overall(D[ds]["genus_abs"],
                                               D[ds]["genus_rel"], gen_top)

    fig = plt.figure(figsize=(17.6, 7.8))
    gs = fig.add_gridspec(2, 6, hspace=0.85, wspace=0.30,
                          left=0.075, right=0.99, top=0.86, bottom=0.10)
    row_defs = [
        ("Phylum (top 5)",
         [(t, m, 4, False) for t, m in zip(ph_taxa, mats_ph)] +
         [("Top 5 (overall)", mat_ph_all, n_ph_all, False)]),
        ("Genus (top 5)",
         [(t, m, 4, True) for t, m in zip(gen_taxa, mats_gen)] +
         [("Top %d (overall)" % CORR_TOP_N, mat_gen_all, n_gen_all, False)]),
    ]
    for ri, (grp, items) in enumerate(row_defs):
        for ci, (name, mat, n, ital) in enumerate(items):
            ax = fig.add_subplot(gs[ri, ci])
            plot_confusion_facet(ax, mat, n, name, show_ytick=(ci == 0),
                                 italic=ital)
            cons = float(np.trace(mat)) / max(n, 1) * 100.0
            ax.text(1.5, -0.42, "consistency = %.1f%%" % cons, ha="center",
                    va="top", fontsize=9, color="#D43F3A", fontweight="bold")
        fig.axes[ri * 6].text(-0.20, 1.16, grp,
                              transform=fig.axes[ri * 6].transAxes,
                              fontsize=14, fontweight="bold", ha="left",
                              va="bottom")
    panel_letter(fig.axes[0], letter, x=-0.20, y=1.80, fontsize=20)
    fig.text(0.53, 0.03, "Relative abundance (RA class)", ha="center",
             fontsize=12)
    fig.text(0.008, 0.48, "Absolute abundance (AA class)", rotation=90,
             va="center", fontsize=12)
    fig.suptitle("%s: per-taxon AA vs RA Spearman consistency "
                 "(BH adj. p < 0.05; n = 4 pairs per taxon; last column = "
                 "all pairwise within row group)" %
                 C.DATASETS[ds], fontsize=15, fontweight="bold", y=0.975)
    return fig


# Analysis note: see README and original laboratory scripts for methodological details.
PREVALENCE_MIN = 0.30


def full_genus_consistency(D, meta, min_prev=PREVALENCE_MIN):
    """Research analysis helper. See README for pipeline and methodology."""
    rows, mats = [], {}
    for ds in ("16S", "ITS"):
        ga, gr = D[ds]["genus_abs"], D[ds]["genus_rel"]
        common = ga.index.intersection(gr.index)
        ga, gr = ga.loc[common], gr.loc[common]
        comp = meta.loc[common, "compartment"]
        for c in C.COMP_ORDER:
            idx = list(comp[comp == c].index)
            if len(idx) < 10:
                continue
            ga_s, gr_s = ga.loc[idx], gr.loc[idx]
            prev = (gr_s > 0).sum(axis=0) / float(len(idx))
            keep = list(prev[prev >= min_prev].index)
            if len(keep) < 3:
                continue
            ga_s, gr_s = ga_s[keep], gr_s[keep]
            n_g = len(keep)
            iu = np.triu_indices(n_g, k=1)
            n_total = len(iu[0])
            ra = np.empty(n_total); pa = np.empty(n_total)
            rr = np.empty(n_total); pr = np.empty(n_total)
            va, vr = ga_s.values, gr_s.values
            for k, (ii, jj) in enumerate(zip(iu[0], iu[1])):
                ra[k], pa[k] = spearmanr(va[:, ii], va[:, jj])
                rr[k], pr[k] = spearmanr(vr[:, ii], vr[:, jj])
            ra, pa = np.nan_to_num(ra, nan=0.0), np.nan_to_num(pa, nan=1.0)
            rr, pr = np.nan_to_num(rr, nan=0.0), np.nan_to_num(pr, nan=1.0)
            qa, qr = U.bh_q(pa), U.bh_q(pr)
            mat = np.zeros((3, 3))
            for k in range(n_total):
                mat[CLASS_IDX[_cls(ra[k], qa[k])],
                    CLASS_IDX[_cls(rr[k], qr[k])]] += 1
            consistent = float(np.trace(mat))
            flipped = mat[CLASS_IDX["Pos"], CLASS_IDX["Neg"]] + \
                mat[CLASS_IDX["Neg"], CLASS_IDX["Pos"]]
            rows.append({
                "dataset": ds, "compartment": c,
                "n_samples": len(idx), "n_genera": n_g, "n_pairs": n_total,
                "AA_pos": int(mat[0, :].sum()),
                "AA_neg": int(mat[1, :].sum()),
                "AA_ns": int(mat[2, :].sum()),
                "RA_pos": int(mat[:, 0].sum()),
                "RA_neg": int(mat[:, 1].sum()),
                "RA_ns": int(mat[:, 2].sum()),
                "consistent": int(consistent),
                "flipped": int(flipped),
                "consistent_ratio": consistent / max(n_total, 1),
            })
            mats[(ds, c)] = (mat, n_total)
    return pd.DataFrame(rows), mats


def make_full_consistency_plate(full_df, mats, letter="F"):
    """Research analysis helper. See README for pipeline and methodology."""
    fig = plt.figure(figsize=(16.5, 7.8))
    gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=0.28,
                          left=0.075, right=0.99, top=0.86, bottom=0.115)
    for ri, ds in enumerate(("16S", "ITS")):
        for ci, c in enumerate(C.COMP_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            mat, n = mats[(ds, c)]
            row = full_df[(full_df["dataset"] == ds) &
                          (full_df["compartment"] == c)].iloc[0]
            name = "%s\n(n = %d pairs)" % (C.COMP_LABELS[ci], n)
            plot_confusion_facet(ax, mat, n, name, show_ytick=(ci == 0))
            ax.text(1.5, -0.42, "consistency = %.1f%%" %
                    (row["consistent_ratio"] * 100), ha="center",
                    va="top", fontsize=9, color="#D43F3A",
                    fontweight="bold")
        fig.axes[ri * 6].text(-0.20, 1.52, C.DATASETS[ds],
                              transform=fig.axes[ri * 6].transAxes,
                              fontsize=14, fontweight="bold", ha="left",
                              va="bottom")
    panel_letter(fig.axes[0], letter, x=-0.20, y=1.95, fontsize=20)
    fig.text(0.53, 0.03, "Relative abundance (RA class)", ha="center",
             fontsize=12)
    fig.text(0.008, 0.48, "Absolute abundance (AA class)", rotation=90,
             va="center", fontsize=12)
    fig.suptitle("Full-genus pairwise AA vs RA Spearman consistency "
                 "(prevalence >= %.0f%%, BH adj. p < 0.05)" %
                 (PREVALENCE_MIN * 100),
                 fontsize=15, fontweight="bold", y=0.975)
    return fig


# Analysis note: see README and original laboratory scripts for methodological details.
def write_dilution_table(D, meta):
    all_rows = []
    for ds in ("16S", "ITS"):
        d = D[ds]
        for lvl in ("phylum", "genus"):
            pa = d["%s_abs" % lvl]
            pr_ = d["%s_rel" % lvl]
            comp = meta.loc[pa.index, "compartment"]
            ma = med_comp(pa, comp)
            mr = med_comp(pr_, comp)
            dil = identify_dilution(ma, mr)
            dil["dataset"] = ds
            dil["level"] = lvl
            if lvl == "genus":
                dil["functional_group"] = dil["taxon"].map(
                    lambda x: PGPR_GENERA.get(x, ""))
            all_rows.append(dil)
    return pd.concat(all_rows, ignore_index=True)


# ========================================================= Main ====
def main():
    for dd in (C.FIGS, C.TABLES, C.TEXT):
        os.makedirs(dd, exist_ok=True)

    print("[sec2] Loading data...")
    meta = load_meta()
    D = load_all(meta)
    print("[sec2] Data loaded: 16S=%d, ITS=%d samples" % (
        len(D["16S"]["phylum_abs"]), len(D["ITS"]["phylum_abs"])))

    # ---- §2.1 dilution ----
    print("[sec2] Identifying dilution taxa...")
    dil_df = write_dilution_table(D, meta)
    dil_df.to_csv(os.path.join(C.TABLES, "tab_sec2_dilution.tsv"),
                  sep="\t", index=False, float_format="%.6g")
    n_dil_phylum = len(dil_df[(dil_df["level"] == "phylum")
                               & (dil_df["dilution_score"] > 0)])
    n_dil_genus = len(dil_df[(dil_df["level"] == "genus")
                              & (dil_df["dilution_score"] > 0)])
    print("[sec2] Dilution taxa: %d phyla, %d genera with positive score" % (
        n_dil_phylum, n_dil_genus))

    # ---- §2.2 Procrustes ----
    print("[sec2] Running Procrustes analysis...")
    proc_rows, proc_coords = run_procrustes_all(D, meta)
    proc_df = pd.DataFrame(proc_rows)
    proc_df.to_csv(os.path.join(C.TABLES, "tab_sec2_procrustes.tsv"),
                   sep="\t", index=False, float_format="%.6g")
    for _, r in proc_df.iterrows():
        print("  %s %s %s: M²=%.4f, r=%.4f, P=%.4f (n=%d)" % (
            r["dataset"], r["scope"], r["region"],
            r["M2"], r["r"], r["p"], r["n"]))

    # ---- §2.3 Spearman ----
    print("[sec2] Computing genus-level Spearman correlations...")
    corr_d = genus_correlations(D, meta)
    print("[sec2] Computing correlation consistency...")
    cons_df = corr_consistency(D, meta, corr_d)
    cons_df.to_csv(os.path.join(C.TABLES, "tab_sec2_corr_summary.tsv"),
                   sep="\t", index=False, float_format="%.6g")
    print("[sec2] Correlation consistency:")
    print(cons_df.to_string(index=False))

    # Analysis note: see README and original laboratory scripts for methodological details.
    print("[sec2] Generating 5 separate plates (A-E)...")
    gen16 = select_top_genera_overall(D["16S"]["genus_rel"], meta, 5)
    genI = select_top_genera_overall(D["ITS"]["genus_rel"], meta, 5)
    phy16 = select_top_genera_overall(D["16S"]["phylum_rel"], meta, 5)
    phyI = select_top_genera_overall(D["ITS"]["phylum_rel"], meta, 5)
    print("[sec2] Top5 bacteria genera:", gen16)
    print("[sec2] Top5 fungi genera:", genI)
    print("[sec2] Top5 bacteria phyla:", phy16)
    print("[sec2] Top5 fungi phyla:", phyI)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for letter, ds, taxa in (("A", "16S", gen16), ("B", "ITS", genI)):
        fig = make_trajectory_plate(D[ds]["genus_abs"], D[ds]["genus_rel"],
                                    meta, taxa, letter,
                                    "%s: top 5 genera along the soil-plant "
                                    "continuum by region" % C.DATASETS[ds])
        fig.savefig(os.path.join(C.FIGS, "fig_sec2_%s.png" % letter),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[sec2] Figure: figs/fig_sec2_%s.png" % letter)

    # Analysis note: see README and original laboratory scripts for methodological details.
    for letter, ds, taxa in (("a", "16S", phy16), ("b", "ITS", phyI)):
        fig = make_trajectory_plate(D[ds]["phylum_abs"], D[ds]["phylum_rel"],
                                    meta, taxa, letter,
                                    "%s: top 5 phyla along the soil-plant "
                                    "continuum by region" % C.DATASETS[ds],
                                    italic_taxa=False, letter_upper=False)
        fig.savefig(os.path.join(C.FIGS, "fig_sec2_S3%s.png" % letter),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[sec2] Figure: figs/fig_sec2_S3%s.png" % letter)

    # ---- Plate C: PCoA (QMP vs RMP) ----
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    proc_overall = proc_df[proc_df["scope"] == "overall"]
    for j, ds in enumerate(("16S", "ITS")):
        plot_pcoa_method(axes[j], proc_coords[ds], ds, "")
        prow = proc_overall[proc_overall["dataset"] == ds].iloc[0]
        axes[j].text(0.97, 0.97, "M\u00b2 = %.3f\nr = %.3f\nP = %.3f" % (
            prow["M2"], prow["r"], prow["p"]),
            transform=axes[j].transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))
    panel_letter(axes[0], "C", fontsize=18)
    fig.tight_layout(w_pad=2.5)
    fig.text(0.5, -0.015,
             "Grey lines link each sample's QMP point to its Procrustes-aligned "
             "RMP point; longer lines = larger per-sample displacement.",
             ha="center", fontsize=9, style="italic", color="#666666")
    fig.savefig(os.path.join(C.FIGS, "fig_sec2_C.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("[sec2] Figure: figs/fig_sec2_C.png")

    # Analysis note: see README and original laboratory scripts for methodological details.
    for letter, ds, phs, gens in (("D", "16S", phy16, gen16),
                                  ("E", "ITS", phyI, genI)):
        fig = make_confusion_plate(ds, phs, gens, meta, D, letter)
        fig.savefig(os.path.join(C.FIGS, "fig_sec2_%s.png" % letter),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[sec2] Figure: figs/fig_sec2_%s.png" % letter)

    # Analysis note: see README and original laboratory scripts for methodological details.
    print("[sec2] Computing full-genus pairwise consistency "
          "(consistency heatmap, prevalence >= %.0f%%)..." % (PREVALENCE_MIN * 100))
    full_df, full_mats = full_genus_consistency(D, meta)
    full_df.to_csv(os.path.join(C.TABLES, "tab_sec2_corr_full.tsv"),
                   sep="\t", index=False, float_format="%.6g")
    for _, r in full_df.iterrows():
        print("  %s %s: %d genera, %d pairs, consistency = %.1f%% "
              "(flipped = %d)" % (r["dataset"], r["compartment"],
                                  r["n_genera"], r["n_pairs"],
                                  r["consistent_ratio"] * 100,
                                  r["flipped"]))
    fig = make_full_consistency_plate(full_df, full_mats)
    fig.savefig(os.path.join(C.FIGS, "fig_sec2_corr_full.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[sec2] Figure: figs/fig_sec2_corr_full.png")

    # Analysis note: see README and original laboratory scripts for methodological details.
    subset_defs = [
        ("S/L/F aboveground", ["S", "L", "F"], "fig_sec2_pcoa_aboveground.png"),
        ("BS/RS/R belowground", ["BS", "RS", "R"], "fig_sec2_pcoa_belowground.png"),
    ]
    for tag, comps, fname in subset_defs:
        print("[sec2] Generating %s PCoA figure..." % tag)
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
        for j, ds in enumerate(("16S", "ITS")):
            pa = D[ds]["phylum_abs"]
            pr_ = D[ds]["phylum_rel"]
            idx = list(meta[meta["compartment"].isin(comps)].index)
            coords_s = pcoa_coords_subset(pa, pr_, idx)
            m2s, r_s, p_s = procrustes_test(coords_s["abs"], coords_s["rel"],
                                            n_perm=499)
            ax = axes[j]
            plot_pcoa_method(ax, coords_s, ds, "AB"[j],
                             title="%s (%s, n = %d)" % (C.DATASETS[ds], tag,
                                                        len(idx)))
            ax.text(0.97, 0.97, "M\u00b2 = %.3f\nr = %.3f\nP = %.3f" % (
                m2s, r_s, p_s),
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))
        fig.tight_layout(w_pad=2.5)
        fig.text(0.5, -0.015,
                 "Grey lines link each sample's QMP point to its Procrustes-aligned "
                 "RMP point; longer lines = larger per-sample displacement.",
                 ha="center", fontsize=9, style="italic", color="#666666")
        fig.savefig(os.path.join(C.FIGS, fname), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        print("[sec2] Figure: figs/%s" % fname)

    # Analysis note: see README and original laboratory scripts for methodological details.
    print("[sec2] Generating regional Procrustes figure (Fig. S7)...")
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 15.2))
    for i, reg in enumerate(C.REGIONS):
        for j, ds in enumerate(("16S", "ITS")):
            pa = D[ds]["phylum_abs"]
            pr_ = D[ds]["phylum_rel"]
            idx = list(meta[meta["region"] == reg].index)
            coords_s = pcoa_coords_subset(pa, pr_, idx)
            m2s, r_s, p_s = procrustes_test(coords_s["abs"], coords_s["rel"],
                                            n_perm=499)
            ax = axes[i, j]
            plot_pcoa_method(ax, coords_s, ds, "abcdef"[i * 2 + j],
                             title="%s (%s, n = %d)" % (C.DATASETS[ds], reg,
                                                        len(idx)),
                             letter_upper=False)
            ax.text(0.97, 0.97, "M\u00b2 = %.3f\nr = %.3f\nP = %.3f" % (
                m2s, r_s, p_s),
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))
    fig.tight_layout(w_pad=2.5, h_pad=3.0)
    fig.text(0.5, -0.005,
             "Grey lines link each sample's QMP point to its Procrustes-aligned "
             "RMP point; longer lines = larger per-sample displacement.",
             ha="center", fontsize=9, style="italic", color="#666666")
    fig.savefig(os.path.join(C.FIGS, "fig_sec2_S7.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("[sec2] Figure: figs/fig_sec2_S7.png")

    # Analysis note: see README and original laboratory scripts for methodological details.
    print("\n" + "=" * 64)
    print("Section 2 analysis complete:")
    for lt in "ABCDE":
        print("  figs/fig_sec2_%s.png" % lt)
    print("  figs/fig_sec2_S3a.png / fig_sec2_S3b.png (phylum-level, Fig. S3)")
    print("  figs/fig_sec2_pcoa_aboveground.png (S/L/F)")
    print("  figs/fig_sec2_pcoa_belowground.png (BS/RS/R)")
    print("  figs/fig_sec2_S7.png (regional Procrustes, WH/SH/SX)")
    print("  tables/tab_sec2_procrustes.tsv  (%d rows)" % len(proc_df))
    print("  tables/tab_sec2_dilution.tsv    (%d taxa)" % len(dil_df))
    print("  tables/tab_sec2_corr_summary.tsv (%d rows)" % len(cons_df))
    print("  tables/tab_sec2_corr_full.tsv   (%d rows)" % len(full_df))
    print("=" * 64)


if __name__ == "__main__":
    main()
