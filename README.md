# Kiwifruit soil-to-fruit microbiome analyses

This repository contains **curated research code** for abundance-resolved bacterial and fungal community analyses in three orchards, six compartments, and the matched fruit metabolome.

## Analysis scripts by manuscript result

| Analysis | Main code | Publication figure |
| --- | --- | --- |
| QMP versus RMP and taxonomic trajectories | `compare_quantitative_and_relative_microbiome_profiles.py`, `analyze_microbiome_taxonomic_composition.py`, `calculate_multifactor_permanova.R` | Figure 1 |
| Marker-gene loads, Shannon diversity and adjacent turnover | `calculate_rarefied_alpha_diversity.py`, `analyze_microbiome_alpha_beta_diversity.py` | Figure 2 |
| Community typing and FEAST source attribution | `analyze_community_assembly_and_source_tracking.R` | Figure 3a–b |
| iCAMP with 999 null-model randomizations | `run_icamp_999_randomizations.R`, `launch_icamp_999_randomizations.ps1`, `summarize_icamp_assembly_999_randomizations.R` | Figure 3c–d |
| Assembly environmental and spatial supplements | `analyze_icamp_environment_and_spatial_patterns.R`, `refresh_icamp_integration_tables.py`, `prepare_soil_climate_environmental_metadata.py` | Supplementary analyses |
| Indicator genera | `calculate_indicator_genera.R` | Supplementary analysis |
| Below/aboveground associations and keystone/robustness summaries | `infer_microbiome_cooccurrence_networks.py`, `analyze_network_topology_and_robustness.py`, `plot_microbial_network_analysis_panels.py` | Figure 4 |
| Spatial occurrence categories | `analyze_microbiome_colonization_patterns.py` | Figure 5 |
| Fruit metabolomics and microbial associations | `analyze_fruit_metabolome_microbiome_associations.py`, `analyze_fruit_specific_microbiome_roles.py`, `analyze_fruit_metabolome_modules_and_source_tracking.py`, `calculate_fruit_metabolome_additional_statistics.py` | Figure 6 / supplements |
| Fruit microbial network, MEpink and exploratory prediction | `infer_fruit_bacteria_fungi_cooccurrence_network.py`, `identify_core_fruit_microbiome_taxa.py`, `track_mepink_metabolite_module_across_compartments.py` | Figure 6 |

Shared code: `analysis_configuration.py`, `analysis_helpers.py`. Other related scripts perform more detailed network and figure computations.

## Data and execution

1. Install dependencies in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
2. Prepare inputs as documented in [`docs/INPUTS.md`](docs/INPUTS.md). This ZIP does **not** include raw reads, input abundance tables, metabolomics features, or the precomputed WGCNA assignments.
3. Set `KIWI_ROOT` to the project **data root**. Python scripts are run from the repository directory and use `analysis_configuration.py`; R scripts read inputs relative to `KIWI_ROOT`. The optional environment variables `KIWI_AMPLICON_DIR` and `KIWI_METABOLOMICS_DIR` override those input folders.
4. Run upstream data preparation before downstream analyses, and inspect outputs before launching figure scripts. The 999-randomization iCAMP launcher may require substantial memory/CPU.

## Sample terminology

| Sequencing source code | Analysis code | Text/figure site | Geographic site |
| --- | --- | --- | --- |
| WH | WH | WC | Wuchang |
| BZ | SH | JS | Jinshan |
| SX | SX | BJ | Baoji |

Compartments: BS (bulk soil), RS (rhizosphere), R (root), S (stem), L (leaf), F (fruit). The original data contain an `X_S_8` label normalized to `SX_S_8` by the analysis code; **confirm this alias with the sequencing laboratory**.

## Scientific limitations of this code release

- Scripts have been reviewed for manuscript relevance and Python syntax, but **have not been rerun on original data** in this environment. Numerical equality with paper results has not been verified.
- The upstream spike-in normalization/ASV construction pipeline and the original **WGCNA module fitting and module-stability resampling** scripts are not present in the uploaded source. Downstream scripts read the WGCNA module assignment table; that is not equivalent to reproducing module construction.
- `calculate_indicator_genera.R` is a publication-aligned *new clean implementation* of IndVal.g with BH correction. Confirm its results against the original indicator table before treating it as the script used for the final figure.
- Some scripts rely on original vendor data labels, legacy result-table names, or intermediate outputs. Do not rename raw data or interpret source tracking as direct transmission.
- R and PowerShell execution were not tested in this environment; see `docs/RELEASE_CHECKLIST.md`.

No archived exploratory scripts or manuscript-writing materials are included in this public folder.
