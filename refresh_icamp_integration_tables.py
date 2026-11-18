# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os
import shutil

import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.environ.get("KIWI_ROOT", os.path.dirname(__file__)))
TABLES = os.path.join(ROOT, "result", "tables")
BAK = os.path.join(ROOT, "result", "r_out", "pre_refresh_backup")


def main():
    if not os.path.isdir(BAK):
        os.makedirs(BAK)
    for ds in ("16S", "ITS"):
        ic = pd.read_csv(os.path.join(TABLES, "tab_eco_icamp_%s.tsv" % ds),
                         sep="\t")
        det = (ic[ic.Category == "Deterministic"]
               .groupby("Compartment")["Proportion"].mean())

        for fn in ("tab_sec15_strategy_icamp_%s.tsv" % ds,
                   "tab_sec15_integration_%s.tsv" % ds):
            path = os.path.join(TABLES, fn)
            bak = os.path.join(BAK, fn)
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
                print("  backup ->", bak)
            df = pd.read_csv(path, sep="\t", index_col=0)
            df["deterministic_frac"] = det.reindex(df.index).values
            df.to_csv(path, sep="\t")
            print("  refreshed", fn)

        # Analysis note: see README and original laboratory scripts for methodological details.
        si = pd.read_csv(os.path.join(
            TABLES, "tab_sec15_strategy_icamp_%s.tsv" % ds), sep="\t")
        r, p = spearmanr(si.colonizer_frac, si.deterministic_frac)
        print("[%s] rho(colonizer, deterministic) = %.4f, P = %.4f, n = %d"
              % (ds, r, p, len(si)))


if __name__ == "__main__":
    main()
