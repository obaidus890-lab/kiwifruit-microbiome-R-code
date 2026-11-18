# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import numpy as np
import pandas as pd

import analysis_configuration as C

RNG = np.random.default_rng(20260812)
TAX_COLS = ["Taxonomy", "superkingdom", "phylum", "class", "order",
            "family", "genus", "species"]


def rarefy_counts(c, depth, rng):
    '\u7cbe\u786e\u5e8f\u8d2f\u6761\u4ef6\u8d85\u51e0\u4f55\u7a00\u7f55\u5316\uff1ac \u4e3a\u975e\u96f6\u8ba1\u6570\u5411\u91cf\uff0c\u8fd4\u56de\u7a00\u7f55\u5316\u8ba1\u6570\u3002'
    colors = c.astype(np.int64)
    total = int(colors.sum())
    draw = int(depth)
    out = np.zeros(len(colors), dtype=np.int64)
    for i in range(len(colors)):
        if draw <= 0:
            break
        if i == len(colors) - 1:
            k = draw
        else:
            k = rng.hypergeometric(colors[i], total - colors[i], draw)
        out[i] = k
        draw -= int(k)
        total -= int(colors[i])
    return out


def shannon_h(r):
    r = r[r > 0]
    p = r / r.sum()
    return float(-np.sum(p * np.log(p)))


def main():
    meta = pd.read_csv(os.path.join(C.R_IN, "sample_meta.tsv"), sep="\t")
    meta["region"] = meta["region"].replace(C.REG_MAP)
    meta = meta.set_index("sample_id")

    for ds in ("16S", "ITS"):
        raw = pd.read_csv(os.path.join(C.ROOT, ds, "otu.xls"),
                          sep="\t", index_col=0)
        raw = raw.drop(columns=[c for c in TAX_COLS if c in raw.columns])
        raw.columns = ["SX_S_8" if c == "X_S_8" else c for c in raw.columns]
        smp = [s for s in meta.index if s in raw.columns]
        if len(smp) != len(meta):
            raise RuntimeError('%s \u6837\u672c\u7f3a\u5931: %d/%d' % (ds, len(smp), len(meta)))
        mat = raw[smp].astype(float)

        tot = mat.sum(axis=0)
        depth = int(tot.min())
        print("[%s] samples=%d, rarefy depth=%d (min reads)" %
              (ds, len(smp), depth), flush=True)

        rows = []
        for s in smp:
            c = mat[s].values
            nz = c > 0
            r_nz = rarefy_counts(c[nz], depth, RNG)
            r = np.zeros(len(c))
            r[nz] = r_nz
            rows.append({
                "sample_id": s,
                "Shannon": shannon_h(r),
                "Observed": int((r > 0).sum()),
                "region": meta.loc[s, "region"],
                "compartment": meta.loc[s, "compartment"],
            })
        df = pd.DataFrame(rows)
        out = os.path.join(C.DATA, "alpha9_%s.tsv" % ds)
        df.to_csv(out, sep="\t", index=False, float_format="%.6g")
        print("[%s] wrote %s (%d rows); Shannon median=%.3f range=[%.3f, %.3f]"
              % (ds, out, len(df), df["Shannon"].median(),
                 df["Shannon"].min(), df["Shannon"].max()), flush=True)


if __name__ == "__main__":
    main()
