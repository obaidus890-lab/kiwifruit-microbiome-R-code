# Rebuild permanova_multifactor.tsv (deleted with amplicon-result dir).
# Genus-level Bray-Curtis PERMANOVA: comm ~ compartment * region, by terms.
# matrix = absolute (genus_abs, copies/g) / relative (genus_rel, reads prop).
ROOT <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
suppressPackageStartupMessages(library(vegan))

meta <- read.delim(file.path(ROOT, "result/r_in/sample_meta.tsv"),
                   stringsAsFactors = FALSE)
meta$region[meta$region == "BZ"] <- "SH"
meta$compartment <- factor(meta$compartment, levels = c("BS", "RS", "R", "S", "L", "F"))
meta$region      <- factor(meta$region, levels = c("WH", "SH", "SX"))
rownames(meta)   <- meta$sample_id

rows <- list()
for (ds in c("16S", "ITS")) {
  for (m in c("absolute", "relative")) {
    fn <- if (m == "absolute") sprintf("genus_abs_%s.tsv", ds) else sprintf("genus_rel_%s.tsv", ds)
    g <- read.delim(file.path(ROOT, "result/data", fn), check.names = FALSE,
                    row.names = 1, stringsAsFactors = FALSE)
    g <- as.matrix(g)
    g <- g[intersect(rownames(g), meta$sample_id), , drop = FALSE]
    mm <- meta[rownames(g), , drop = FALSE]
    dm <- vegdist(g, method = "bray")
    set.seed(20260812)
    ad <- adonis2(dm ~ compartment * region, data = mm, by = "terms", permutations = 999)
    for (trm in rownames(ad)) {
      rows[[length(rows) + 1]] <- data.frame(
        matrix = m, dataset = ds, term = trm,
        Df = ad[trm, "Df"], F = ad[trm, "F"], R2 = ad[trm, "R2"],
        p = ad[trm, "Pr(>F)"], stringsAsFactors = FALSE)
    }
    message(sprintf("[%s %s] n=%d taxa=%d done", ds, m, nrow(g), ncol(g)))
  }
}
tab <- do.call(rbind, rows)
write.table(tab, file.path(ROOT, "result/tables/permanova_multifactor.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(tab, digits = 4)
message("-> result/tables/permanova_multifactor.tsv")
