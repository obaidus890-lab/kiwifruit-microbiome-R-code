# Required data inputs

The following paths are relative to `KIWI_ROOT` unless an environment variable overrides the location.

- `result/r_in/sample_meta.tsv`: 162 sample IDs with `region` and `compartment` (and `replicate`, `plant` where required).
- `16S/otu.xls` and `ITS/otu.xls`: ASV read-count matrices for rarefaction; check that they contain all samples.
- `16S/unit_sample_otu_copies.xls`, `ITS/unit_sample_otu_copies.xls`: spike-in-adjusted quantitative ASV estimates used by Python loaders.
- `pre/{16S,ITS}/Analysis/G1-Compare_All/Absolute_Quantitation/OTU/otu_copies.txt`: quantitative ASV matrix for R iCAMP and source tracking.
- `pre/{16S,ITS}/Analysis/G1-Compare_All/Absolute_Quantitation/OTU/rep.seq.tre`: ASV phylogenetic tree.
- `result/data/genus_abs_{16S,ITS}.tsv`, `result/data/genus_rel_{16S,ITS}.tsv`: genus matrices for statistical analyses.
- `result/data/sample_loads.tsv`, `result/data/alpha9_{16S,ITS}.tsv`: load and alpha-diversity inputs / derivatives.
- `metabolomics/metabolome.csv`, `metabolomics/group_info.csv`: matched fruit metabolomics tables (or set `KIWI_METABOLOMICS_DIR`).
- `metabolomics/analysis_output/WGCNA/WGCNA_module_assignment.csv`: **precomputed** assignments required by downstream metabolome scripts. Original fitting/stability code is not available here.
- Soil/climate source tables and intermediate `result/r_in` and `result/tables` datasets as referenced by each script.

## Important normalization

The analysis uses WH→WC (Wuchang), BZ→SH→JS (Jinshan), and SX→BJ (Baoji). `X_S_8` is an unconfirmed source filename alias for `SX_S_8`; verify before data publication. Align all 162 microbiome samples and 27 fruit samples using the same plant and replicate IDs.

Some original source data use non-English column headings and folder names. Their published data dictionary/translation must be completed to make the full pipeline runnable with an English-only input tree. **Do not merely translate a column name in code without translating the corresponding source input.**
