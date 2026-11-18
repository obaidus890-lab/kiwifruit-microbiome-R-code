# Release verification checklist

- [ ] Every raw and processed input is available at the path documented in INPUTS.md.
- [ ] The scripts execute in a fresh environment with the documented package versions.
- [ ] The WGCNA fitting and module stability scripts (or a documented reproducibility alternative) are deposited.
- [ ] Spike-in quantification / ASV processing upstream scripts are deposited or linked.
- [ ] Run `calculate_indicator_genera.R` with the original tables and compare BH-corrected results with the manuscript.
- [ ] Confirm no earlier 99/49-randomization iCAMP results are used in final summary tables.
- [ ] Verify iCAMP outputs show `rand=999` and `bin.size.limit=12` for all 12 runs.
- [ ] Rebuild Figures 1–6 and cited Supplementary Tables/Figures, and compare all numbers and labels.
- [ ] Resolve `X_S_8` versus `SX_S_8` with the sequencing laboratory.
- [ ] Confirm code paths and non-English source-column translations match the deposited data.
- [ ] Record the software versions, seeds, and checksums of final analytical outputs.
