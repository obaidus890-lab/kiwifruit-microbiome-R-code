#!/usr/bin/env Rscript
# Genus-level indicator-value analysis for compartment and orchard.
# Input files: result/r_in/sample_meta.tsv, result/r_in/genus_abs_{16S,ITS}.tsv.
# Uses IndVal.g and 999 permutations. BH correction is applied within each
# kingdom x grouping family, consistent with the study Methods.
suppressPackageStartupMessages({
  library(indicspecies)
  library(permute)
})
root <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
input_dir <- file.path(root, "result", "r_in")
output_dir <- file.path(root, "result", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
meta <- read.delim(file.path(input_dir, "sample_meta.tsv"), check.names = FALSE)
meta$region <- ifelse(meta$region == "BZ", "SH", meta$region)
set.seed(42)
all_results <- list()
for (ds in c("16S", "ITS")) {
  data_path <- file.path(input_dir, sprintf("genus_abs_%s.tsv", ds))
  # A second supported input location is result/data/genus_abs_{ds}.tsv.
  if (!file.exists(data_path)) data_path <- file.path(root, "result", "data", sprintf("genus_abs_%s.tsv", ds))
  comm <- read.delim(data_path, check.names = FALSE, row.names = 1)
  sample_ids <- intersect(rownames(comm), meta$sample_id)
  if (length(sample_ids) != nrow(meta)) stop("Sample names are missing from genus table: ", ds)
  comm <- as.matrix(comm[meta$sample_id, , drop = FALSE])
  storage.mode(comm) <- "numeric"
  comm <- comm[, colSums(comm, na.rm = TRUE) > 0, drop = FALSE]
  if (any(!is.finite(comm)) || any(comm < 0)) stop("Invalid values in genus table: ", ds)
  for (grouping in c("compartment", "region")) {
    labels <- factor(meta[[grouping]])
    # duleg=TRUE limits significant patterns to individual, rather than combined, groups.
    fit <- indicspecies::multipatt(comm, labels, func = "IndVal.g",
                                  duleg = TRUE, control = permute::how(nperm = 999))
    result <- fit$sign
    result$genus <- rownames(result)
    result$dataset <- ds
    result$grouping <- grouping
    result$q_value <- p.adjust(result$p.value, method = "BH")
    all_results[[paste(ds, grouping)]] <- result
    write.table(result, file.path(output_dir, sprintf("indicator_genera_%s_%s.tsv", ds, grouping)),
                sep = "\t", quote = FALSE, row.names = FALSE)
    message(sprintf("%s / %s: %d genera; FDR < 0.05: %d", ds, grouping,
                    nrow(result), sum(result$q_value < 0.05, na.rm = TRUE)))
  }
}
