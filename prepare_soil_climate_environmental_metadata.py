# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

import analysis_configuration as C
import analysis_helpers as U

# Analysis note: see README and original laboratory scripts for methodological details.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENV_SRC = os.path.join(C.ROOT, '\u73af\u5883\u8d44\u6599')
CLIMATE_CSV = {"WH": os.path.join(ENV_SRC, "WH.csv"),
               "SH": os.path.join(ENV_SRC, "SH.csv"),
               "SX": os.path.join(ENV_SRC, "MX.csv")}       # MX -> SX
ENV_XLSX = os.path.join(ENV_SRC, '\u73af\u5883\u7279\u5f81.xlsx')
OLD_CLIMATE = os.path.join(C.AMP, '04_\u73af\u5883\u56e0\u5b50\u4e0e\u7edd\u5bf9\u4e30\u5ea6\u5173\u8054',
                           "climate_by_region.tsv")

# Analysis note: see README and original laboratory scripts for methodological details.
SOIL_EN = {"pH": "pH", '\u6709\u673a\u8d28(g/kg)': "SOM", '\u5168\u6c2e(g/kg)': "TN",
           '\u6c34\u89e3\u6027\u6c2e(mg/kg)': "HN", '\u94f5\u6001\u6c2e(mg/kg)': "NH4-N",
           '\u4e9a\u785d\u6001\u6c2e(mg/kg)': "NO2-N", '\u5168\u78f7(g/kg)': "TP",
           '\u6709\u6548\u78f7(mg/kg)': "AP", '\u5168\u94be(g/kg)': "TK",
           '\u901f\u6548\u94be(mg/kg)': "AK", '\u8132\u9176(U/g \u571f\u6837)': "Urease",
           '\u8517\u7cd6\u9176(U/g \u571f\u6837)': "Invertase", '\u8fc7\u6c27\u5316\u6c22\u9176(U/g \u571f\u6837)': "Catalase",
           '\u7ea4\u7ef4\u7d20\u9176(U/g \u571f\u6837)': "Cellulase", '\u78f7\u9178\u9176(U/g \u571f\u6837)': "Phosphatase"}

# Analysis note: see README and original laboratory scripts for methodological details.
BG_COMPS, AG_COMPS = ["BS", "RS", "R"], ["S", "L", "F"]


# ---------------------------------------------------------------- climate ----
def pick_col(cols, keyword):
    '\u6309\u5305\u542b\u5173\u952e\u5b57\u5339\u914d\u5217\u540d\uff08\u4e09\u6587\u4ef6\u5217\u5e8f\u4e0d\u540c\uff0c\u5fc5\u987b\u6309\u5217\u540d\u53d6\u6570\uff09\u3002'
    hits = [c for c in cols if keyword in str(c)]
    if len(hits) != 1:
        raise RuntimeError('\u5217\u5339\u914d\u5931\u8d25: keyword=%r hits=%r cols=%r'
                           % (keyword, hits, list(cols)))
    return hits[0]


def load_climate_region(region):
    '\u8bfb\u5355\u5730\u533a\u6708\u5ea6\u6c14\u5019 CSV\uff08utf-8-sig\uff09\uff0c\u8fd4\u56de\u542b month/temp/hum/precip \u7684\u8868\u3002'
    df = pd.read_csv(CLIMATE_CSV[region], encoding="utf-8-sig")
    print('[climate] %s \u5217\u540d: %s' % (region, list(df.columns)), flush=True)
    col_date = pick_col(df.columns, '\u65e5\u671f')
    col_temp = pick_col(df.columns, '2\u7c73\u5e73\u5747\u6c14\u6e29')
    col_hum = pick_col(df.columns, '2\u7c73\u5e73\u5747\u76f8\u5bf9\u6e7f\u5ea6')
    col_pre = pick_col(df.columns, '\u603b\u964d\u6c34\u91cf')
    out = pd.DataFrame({
        "month": [U.month_from_row(s, i + 1)
                  for i, s in enumerate(df[col_date].tolist())],
        "temp": df[col_temp].astype(float),
        "hum": df[col_hum].astype(float),
        "precip": df[col_pre].astype(float)})
    if len(out) != 9 or sorted(out["month"]) != list(range(1, 10)):
        raise RuntimeError('%s \u6c14\u5019\u6708\u4efd\u5f02\u5e38: %s' % (region, out["month"].tolist()))
    return out


def climate_windows():
    '\u751f\u957f\u5b63\uff081-9 \u6708\uff09\u4e0e\u590f\u5b63\uff086-8 \u6708\uff09\u53cc\u7a97\u53e3\u7edf\u8ba1 + \u65b0\u65e7\u5bf9\u7167\u65ad\u8a00\u3002'
    rows, detail = [], {}
    for rg in C.REGIONS:
        m = load_climate_region(rg)
        detail[rg] = m
        gs, sm = m, m[m["month"].between(6, 8)]
        rows.append({"region": rg,
                     "gs_mean_temp": gs["temp"].mean(),
                     "gs_rel_humidity": gs["hum"].mean(),
                     "gs_total_precip": gs["precip"].sum(),
                     "summer_mean_temp": sm["temp"].mean(),
                     "summer_rel_humidity": sm["hum"].mean(),
                     "summer_precip_sum": sm["precip"].sum()})
    win = pd.DataFrame(rows).set_index("region").loc[C.REGIONS]
    win.to_csv(os.path.join(C.TABLES, "tab_14_climate_windows.tsv"),
               sep="\t", float_format="%.6g")

    # Analysis note: see README and original laboratory scripts for methodological details.
    assert 16 <= win.loc["WH", "gs_mean_temp"] <= 21, \
        'WH \u751f\u957f\u5b63\u5747\u6e29\u8d8a\u754c: %.4f' % win.loc["WH", "gs_mean_temp"]
    assert 65 <= win.loc["WH", "gs_rel_humidity"] <= 80, \
        'WH \u751f\u957f\u5b63\u5747\u6e7f\u8d8a\u754c: %.4f' % win.loc["WH", "gs_rel_humidity"]

    # Analysis note: see README and original laboratory scripts for methodological details.
    cmp_rows = []
    if os.path.exists(OLD_CLIMATE):
        old = pd.read_csv(OLD_CLIMATE, sep="\t").set_index("region")
        tol = lambda a, b, eps: abs(a - b) <= eps
        for rg in C.REGIONS:
            assert tol(win.loc[rg, "gs_total_precip"],
                       old.loc[rg, "total_precip"], 0.5), \
                '%s \u603b\u964d\u6c34\u4e0e\u65e7\u8868\u4e0d\u4e00\u81f4: %.2f vs %.2f' % (
                    rg, win.loc[rg, "gs_total_precip"], old.loc[rg, "total_precip"])
        for rg in ("SH", "SX"):
            assert tol(win.loc[rg, "gs_mean_temp"], old.loc[rg, "mean_temp"], 0.05), \
                '%s \u5747\u6e29\u4e0e\u65e7\u8868\u4e0d\u4e00\u81f4' % rg
            assert tol(win.loc[rg, "gs_rel_humidity"], old.loc[rg, "rel_humidity"], 0.05), \
                '%s \u5747\u6e7f\u4e0e\u65e7\u8868\u4e0d\u4e00\u81f4' % rg
        assert abs(win.loc["WH", "gs_mean_temp"] - old.loc["WH", "mean_temp"]) > 1.0, \
            'WH \u65b0\u5747\u6e29\u4e0d\u5e94\u7b49\u4e8e\u65e7\u9519\u503c'
        assert abs(win.loc["WH", "gs_rel_humidity"] - old.loc["WH", "rel_humidity"]) > 1.0, \
            'WH \u65b0\u5747\u6e7f\u4e0d\u5e94\u7b49\u4e8e\u65e7\u9519\u503c'
        for rg in C.REGIONS:
            cmp_rows.append({
                "region": rg,
                "old_mean_temp": old.loc[rg, "mean_temp"],
                "old_rel_humidity": old.loc[rg, "rel_humidity"],
                "old_total_precip": old.loc[rg, "total_precip"],
                "new_mean_temp": win.loc[rg, "gs_mean_temp"],
                "new_rel_humidity": win.loc[rg, "gs_rel_humidity"],
                "new_total_precip": win.loc[rg, "gs_total_precip"]})
    else:
        print('[climate] \u65e7\u8868\u7f3a\u5931\uff08\u6269\u589e\u5b50\u7ed3\u679c\u5df2\u5220\u9664\uff09\uff0c\u8df3\u8fc7\u65b0\u65e7\u5bf9\u7167\u65ad\u8a00',
              flush=True)
        for rg in C.REGIONS:
            cmp_rows.append({
                "region": rg,
                "old_mean_temp": np.nan,
                "old_rel_humidity": np.nan,
                "old_total_precip": np.nan,
                "new_mean_temp": win.loc[rg, "gs_mean_temp"],
                "new_rel_humidity": win.loc[rg, "gs_rel_humidity"],
                "new_total_precip": win.loc[rg, "gs_total_precip"]})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(C.TABLES, "tab_14_climate_old_new.tsv"),
                  sep="\t", index=False, float_format="%.6g")
    print('[assert] \u6c14\u5019\u65ad\u8a00\u5168\u90e8\u901a\u8fc7\uff1a')
    print(cmp_df.to_string(index=False), flush=True)
    return win


# ------------------------------------------------------------------ soil ----
def load_soil():
    '\u571f\u58e4 sheet 9 \u6837\u672c x 15 \u6307\u6807\u82f1\u6587\u5316\uff1b\u8fd4\u56de (\u6837\u672c\u8868, \u5730\u533a\u5747\u503c\u8868)\u3002'
    raw = pd.read_excel(ENV_XLSX, sheet_name='\u571f\u58e4')
    region = raw.iloc[:, 0].ffill().map(
        lambda x: C.REG_MAP.get(str(x), str(x)))
    soil = raw.iloc[:, 1:].astype(float).copy()
    soil.columns = [SOIL_EN.get(c, c) for c in soil.columns]
    if list(soil.columns) != list(SOIL_EN.values()):
        raise RuntimeError('\u571f\u58e4\u5217\u6620\u5c04\u5931\u8d25: %s' % list(soil.columns))
    soil.insert(0, "sample_id", ["%s_%d" % (r, i % 3 + 1)
                                 for i, r in enumerate(region.values)])
    soil.insert(1, "region", region.values)
    soil.to_csv(os.path.join(C.TABLES, "tab_14_soil.tsv"), sep="\t",
                index=False, float_format="%.6g")
    mean = soil.groupby("region")[list(SOIL_EN.values())].mean().loc[C.REGIONS]
    mean.to_csv(os.path.join(C.TABLES, "tab_14_soil_region_mean.tsv"),
                sep="\t", float_format="%.6g")
    print('[soil] 9 \u6837\u672c x 15 \u6307\u6807\u82f1\u6587\u5316\u5b8c\u6210', flush=True)
    return soil, mean


# Analysis note: see README and original laboratory scripts for methodological details.
def build_n18(clim_win, soil_mean):
    """Research analysis helper. See README for pipeline and methodology."""
    loads = pd.read_csv(os.path.join(C.DATA, "sample_loads.tsv"), sep="\t")
    meta = pd.read_csv(C.META_PATH, sep="\t")
    meta["region"] = meta["region"].replace(C.REG_MAP)
    meta = meta.set_index("sample_id")

    env_rows, hell_rows = [], []
    for ds in ("16S", "ITS"):
        alpha = pd.read_csv(
            os.path.join(C.DATA, "alpha9_%s.tsv" % ds),
            sep="\t")
        alpha = alpha.set_index("sample_id")
        gen = pd.read_csv(os.path.join(C.DATA, "genus_abs_%s.tsv" % ds),
                          sep="\t", index_col=0).astype(float)
        lds = loads[loads["dataset"] == ds].set_index("sample_id")

        for comp in C.COMP_ORDER:
            for rg in C.REGIONS:
                smp = [s for s in meta.index
                       if meta.loc[s, "compartment"] == comp
                       and meta.loc[s, "region"] == rg
                       and s in lds.index]
                if not smp:
                    raise RuntimeError('\u7a7a\u5355\u5143: %s %s %s' % (ds, comp, rg))
                cell = "%s_%s_%s" % (ds, comp, rg)
                row = {"cell_id": cell, "dataset": ds,
                       "compartment": comp, "region": rg,
                       "n_samples": len(smp),
                       "log10_load": lds.loc[smp, "log10_load"].mean(),
                       "shannon": alpha.loc[smp, "Shannon"].mean()}
                # Analysis note: see README and original laboratory scripts for methodological details.
                row["mean_temp"] = clim_win.loc[rg, "gs_mean_temp"]
                row["rel_humidity"] = clim_win.loc[rg, "gs_rel_humidity"]
                row["total_precip"] = clim_win.loc[rg, "gs_total_precip"]
                for v in SOIL_EN.values():
                    row[v] = soil_mean.loc[rg, v]
                env_rows.append(row)

                # Analysis note: see README and original laboratory scripts for methodological details.
                gsum = gen.loc[smp].sum(axis=0)
                gsum = gsum[gsum > 0]
                hell_rows.append(pd.Series(np.sqrt(gsum / gsum.sum()),
                                           name=cell))
    env18 = pd.DataFrame(env_rows)
    hell = pd.DataFrame(hell_rows).fillna(0.0)
    return env18, hell


# ---------------------------------------------------------------- AGBG ----
def build_agbg():
    """Research analysis helper. See README for pipeline and methodology."""
    loads = pd.read_csv(os.path.join(C.DATA, "sample_loads.tsv"), sep="\t")
    meta = pd.read_csv(C.META_PATH, sep="\t")
    meta["region"] = meta["region"].replace(C.REG_MAP)
    loads = loads.merge(meta[["sample_id", "region", "compartment", "plant",
                              "replicate", "underground"]],
                        on="sample_id", how="left")
    if loads["compartment"].isna().any():
        raise RuntimeError('loads \u4e0e metadata \u6837\u672c\u4e0d\u5339\u914d')

    def ratio_stat(g):
        bg = g[g["compartment"].isin(BG_COMPS)]["log10_load"]
        ag = g[g["compartment"].isin(AG_COMPS)]["log10_load"]
        if len(bg) == 0 or len(ag) == 0:
            return np.nan
        return bg.mean() - ag.mean()

    rows = []
    for ds in ("16S", "ITS"):
        lds = loads[loads["dataset"] == ds]
        # Analysis note: see README and original laboratory scripts for methodological details.
        # Analysis note: see README and original laboratory scripts for methodological details.
        # Analysis note: see README and original laboratory scripts for methodological details.
        for (rg, rep), g in lds.groupby(["region", "replicate"], sort=True):
            v = ratio_stat(g)
            if np.isnan(v):
                continue
            rows.append({"dataset": ds, "scale": "plant",
                         "plant_id": "%s_%d" % (rg, int(rep)),
                         "group": "%s_%d" % (rg, int(rep)),
                         "region": rg,
                         "n_bg": int(g["compartment"].isin(BG_COMPS).sum()),
                         "n_ag": int(g["compartment"].isin(AG_COMPS).sum()),
                         "log10_bg_load":
                             g[g["compartment"].isin(BG_COMPS)]["log10_load"].mean(),
                         "log10_ag_load":
                             g[g["compartment"].isin(AG_COMPS)]["log10_load"].mean(),
                         "agbg_ratio_log10": v,
                         "agbg_ratio_linear": 10.0 ** v})
        # Analysis note: see README and original laboratory scripts for methodological details.
        for rg in C.REGIONS:
            g = lds[lds["region"] == rg]
            rows.append({"dataset": ds, "scale": "region",
                         "plant_id": "NA", "group": rg, "region": rg,
                         "n_bg": int(g["compartment"].isin(BG_COMPS).sum()),
                         "n_ag": int(g["compartment"].isin(AG_COMPS).sum()),
                         "log10_bg_load":
                             g[g["compartment"].isin(BG_COMPS)]["log10_load"].mean(),
                         "log10_ag_load":
                             g[g["compartment"].isin(AG_COMPS)]["log10_load"].mean(),
                         "agbg_ratio_log10": ratio_stat(g),
                         "agbg_ratio_linear": 10.0 ** ratio_stat(g)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(C.TABLES, "tab_14_agbg.tsv"), sep="\t",
               index=False, float_format="%.6g")
    print('[agbg] plant \u884c=%d, region \u884c=%d'
          % (int((tab["scale"] == "plant").sum()),
             int((tab["scale"] == "region").sum())), flush=True)
    return tab


# ----------------------------------------------------------------- VIF ----
def vif_screen(env18):
    'n=18 \u73af\u5883\u77e9\u9635\uff0815 \u571f\u58e4 + 3 \u6c14\u5019\uff09z \u6807\u51c6\u5316\u540e\u8fed\u4ee3\u5254\u9664 VIF>10\u3002'
    vars18 = list(SOIL_EN.values()) + ["mean_temp", "rel_humidity", "total_precip"]
    X = env18[vars18].astype(float)
    Xz = (X - X.mean()) / (X.std(ddof=0) + 1e-12)

    records, kept = [], list(vars18)
    round_no = 0
    while True:
        mat = Xz[kept].values
        mat_c = np.column_stack([np.ones(mat.shape[0]), mat])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Analysis note: see README and original laboratory scripts for methodological details.
            vifs = {v: variance_inflation_factor(mat_c, i + 1)
                    for i, v in enumerate(kept)}
        worst = max(vifs, key=vifs.get)
        if not np.isfinite(vifs[worst]) or vifs[worst] > 10:
            round_no += 1
            rec = {"round": round_no, "n_vars": len(kept),
                   "dropped": worst, "vif_dropped": vifs[worst]}
            for v in kept:
                rec["vif_%s" % v] = vifs[v]
            records.append(rec)
            kept.remove(worst)
        else:
            round_no += 1
            rec = {"round": round_no, "n_vars": len(kept),
                   "dropped": "none", "vif_dropped": vifs[worst]}
            for v in kept:
                rec["vif_%s" % v] = vifs[v]
            records.append(rec)
            break
    vif_df = pd.DataFrame(records)
    vif_df.to_csv(os.path.join(C.TABLES, "tab_14_vif.tsv"), sep="\t",
                  index=False, float_format="%.6g")
    print('[vif] \u5254\u9664 %d \u4e2a\u53d8\u91cf\uff0c\u6700\u7ec8\u4fdd\u7559 %d \u4e2a: %s'
          % (len(vars18) - len(kept), len(kept), ", ".join(kept)), flush=True)
    return vif_df, kept


# ------------------------------------------------------------- R inputs ----
def write_r_inputs(env18, clim_win, soil_mean):
    'env_n18.tsv\uff08\u7eaf\u73af\u5883 18 \u884c\uff09\u4e0e sample_level.tsv\uff08162 \u884c\uff0c\u4f9b Mantel\uff09\u3002'
    # Analysis note: see README and original laboratory scripts for methodological details.
    full_cols = ["cell_id", "dataset", "compartment", "region",
                 "log10_load", "shannon"] + list(SOIL_EN.values()) + \
                ["mean_temp", "rel_humidity", "total_precip"]
    env18[full_cols].to_csv(
        os.path.join(C.TABLES, "tab_14_env_matrix_n18.tsv"),
        sep="\t", index=False, float_format="%.6g")

    # Analysis note: see README and original laboratory scripts for methodological details.
    # Analysis note: see README and original laboratory scripts for methodological details.
    env_only = env18.drop_duplicates(subset=["compartment", "region"]).copy()
    env_only["cell_id"] = env_only["compartment"] + "_" + env_only["region"]
    env_cols18 = ["cell_id", "compartment", "region"] + \
        list(SOIL_EN.values()) + ["mean_temp", "rel_humidity", "total_precip"]
    env_only[env_cols18].to_csv(os.path.join(C.R_IN, "env_n18.tsv"), sep="\t",
                                index=False, float_format="%.6g")

    # Analysis note: see README and original laboratory scripts for methodological details.
    clim_map = {"mean_temp": clim_win["gs_mean_temp"],
                "rel_humidity": clim_win["gs_rel_humidity"],
                "total_precip": clim_win["gs_total_precip"]}
    meta = pd.read_csv(C.META_PATH, sep="\t")
    meta["region"] = meta["region"].replace(C.REG_MAP)
    sl = meta[["sample_id", "compartment", "region", "underground"]].copy()
    for v, src in clim_map.items():
        sl[v] = sl["region"].map(src)
    for v in SOIL_EN.values():
        sl[v] = sl["region"].map(soil_mean[v])
    if sl[list(SOIL_EN.values()) + ["mean_temp"]].isna().any().any():
        raise RuntimeError('sample_level \u73af\u5883\u6302\u63a5\u5b58\u5728\u7f3a\u5931')
    sl.to_csv(os.path.join(C.R_IN, "sample_level.tsv"), sep="\t",
              index=False, float_format="%.6g")
    print('[r_in] env_n18.tsv %d \u884c; sample_level.tsv %d \u884c'
          % (len(env_only), len(sl)), flush=True)


# ----------------------------------------------------------------- main ----
if __name__ == "__main__":
    for dd in (C.TABLES, C.R_IN):
        os.makedirs(dd, exist_ok=True)

    clim_win = climate_windows()                          # Analysis note: see README and original laboratory scripts for methodological details.
    soil, soil_mean = load_soil()                         # Analysis note: see README and original laboratory scripts for methodological details.
    env18, hell = build_n18(clim_win, soil_mean)          # Analysis note: see README and original laboratory scripts for methodological details.
    hell.index.name = "cell_id"
    hell.to_csv(os.path.join(C.R_IN, "comm_hellinger_n18.tsv"), sep="\t",
                float_format="%.6g")
    print('[n18] \u73af\u5883\u77e9\u9635 %s\uff0818 cells x 2 datasets\uff09; Hellinger \u77e9\u9635 %s'
          % (env18.shape, hell.shape), flush=True)

    build_agbg()                                          # 5. AG:BG
    vif_screen(env18)                                     # Analysis note: see README and original laboratory scripts for methodological details.
    write_r_inputs(env18, clim_win, soil_mean)            # 7.

    print('[done] sec14_env_prep \u5168\u90e8\u4ea7\u51fa\u5b8c\u6210', flush=True)
    sys.exit(0)
