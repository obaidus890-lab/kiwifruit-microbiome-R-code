# ============================================================================
# analyze_community_assembly_and_source_tracking.R
# ============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(DirichletMultinomial)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(ape)
  library(vegan)
  library(FEAST)
  library(ggalluvial)
  library(patchwork)
})

set.seed(20260811)

# ------------------------------------------------------------------ paths ----
ROOT     <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
FIGS     <- file.path(ROOT, "result/figs")
TABLES   <- file.path(ROOT, "result/tables")
META_F   <- file.path(ROOT, "result/r_in/sample_meta.tsv")
OTU_PATH <- c("16S" = file.path(ROOT, "pre/16S/Analysis/G1-Compare_All/Absolute_Quantitation/OTU"),
              "ITS" = file.path(ROOT, "pre/ITS/Analysis/G1-Compare_All/Absolute_Quantitation/OTU"))

COMP_ORDER   <- c("BS", "RS", "R", "S", "L", "F")
COMP_LABEL   <- c(BS = "Bulk soil", RS = "Rhizosphere soil", R = "Root",
                  S = "Stem", L = "Leaf", F = "Fruit")
SITE_ORDER   <- c("WH", "SH", "SX")
SITE_COLORS  <- c(WH = "#3C5488", SH = "#E64B35", SX = "#00A087")
COMP_COLORS  <- c(BS = "#8C6D46", RS = "#BD9A5B", R = "#E8C07D",
                  S = "#7FB069", L = "#3E8E5A", F = "#D43F3A")
TAXA_DROP    <- c("#OTU ID", "OTUsize", "Taxonomy", "superkingdom", "phylum",
                  "class", "order", "family", "genus", "species")

theme_pub <- function() {
  theme_classic(base_size = 8, base_family = "Times New Roman") +
    theme(panel.border = element_rect(fill = NA, colour = "black", linewidth = 0.8),
          axis.text = element_text(colour = "black", family = "Times New Roman"),
          legend.key = element_rect(fill = NA))
}

run_segment <- function(name, expr) {
  message("\n===== ", name, " =====")
  out <- tryCatch(expr, error = function(e) {
    message("[FAIL] ", name, ": ", conditionMessage(e)); NULL
  })
  if (!is.null(out)) message("[OK] ", name)
  out
}

write_tsv <- function(df, name) {
  write.table(df, file.path(TABLES, name), sep = "\t", quote = FALSE,
              row.names = FALSE)
  message("  -> tables/", name)
}

# ------------------------------------------------------------- data load ----
load_dataset <- function(ds) {
  d <- OTU_PATH[[ds]]
  raw <- read.delim(file.path(d, "otu_copies.txt"), check.names = FALSE,
                    stringsAsFactors = FALSE)
  tax_cols <- intersect(TAXA_DROP, names(raw))
  samp_cols <- setdiff(names(raw), tax_cols)

  tax_keep <- intersect(c("phylum", "class", "order", "family",
                         "genus", "species"), names(raw))
  tax_df <- raw[, c(names(raw)[1], tax_keep), drop = FALSE]
  names(tax_df)[1] <- "ASV"

  meta <- read.delim(META_F, stringsAsFactors = FALSE)
  meta$region <- as.character(meta$region)
  meta$region[meta$region == "BZ"] <- "SH"
  meta$compartment <- factor(meta$compartment, levels = COMP_ORDER)
  meta$site        <- factor(meta$region, levels = SITE_ORDER)

  keep <- intersect(samp_cols, meta$sample_id)
  comm <- as.matrix(raw[keep])
  rownames(comm) <- raw[[1]]
  storage.mode(comm) <- "double"
  comm <- round(comm)
  comm <- comm[rowSums(comm) > 0, , drop = FALSE]
  comm <- t(comm)                                # samples x taxa

  meta <- meta[match(rownames(comm), meta$sample_id), , drop = FALSE]
  rownames(meta) <- meta$sample_id

  tree <- tryCatch(read.tree(file.path(d, "rep.seq.tre")), error = function(e) NULL)
  if (!is.null(tree)) {
    keep_tip <- tree$tip.label %in% colnames(comm)
    tree <- drop.tip(tree, tree$tip.label[!keep_tip])
    com <- intersect(colnames(comm), tree$tip.label)
    comm <- comm[, com, drop = FALSE]
  }

  prev <- colSums(comm > 0) / nrow(comm)
  comm <- comm[, prev >= 0.10, drop = FALSE]
  if (!is.null(tree)) tree <- keep.tip(tree, intersect(tree$tip.label, colnames(comm)))

  message(sprintf("[%s] samples=%d  ASVs=%d  tree tips=%s", ds,
                  nrow(comm), ncol(comm),
                  ifelse(is.null(tree), "NA", length(tree$tip.label))))
  list(comm = comm, meta = meta, tree = tree, tax = tax_df)
}

agg_to_genus <- function(comm, tax) {
  asv <- colnames(comm)
  g <- tax$genus[match(asv, tax$ASV)]
  g[is.na(g) | g == "" | grepl("^uncultured", g, ignore.case = TRUE)] <- "g__Unclassified"
  ph <- tax$phylum[match(asv, tax$ASV)]
  g[grepl("^g__Unclassified", g)] <- paste0("g__Unclassified_", ph[grepl("^g__Unclassified", g)])
  agg <- t(rowsum(t(comm), group = g))   # samples x genus
  agg <- agg[, colSums(agg) > 0, drop = FALSE]
  agg
}

# ============================================================================
# ============================================================================
run_dmm <- function(ds, dat) {
  comm_g <- agg_to_genus(dat$comm, dat$tax)
  message(sprintf("[%s] DMM genus-level: %d genera", ds, ncol(comm_g)))
  k_range  <- 1:8
  dmm_fits <- lapply(k_range, function(k) {
    message("  fitting K = ", k, " ...", appendLF = FALSE); flush.console()
    fit <- dmn(comm_g, k = k, verbose = FALSE)
    message(" done")
    fit
  })
  bic_v <- sapply(dmm_fits, BIC)
  n_param <- sapply(k_range, function(k) k * ncol(comm_g) + (k - 1))
  lp_v  <- (n_param * log(nrow(comm_g)) - bic_v) / 2
  best_k <- k_range[which.min(bic_v)]
  message(sprintf("[%s] DMM best K = %d", ds, best_k))

  post <- mixture(dmm_fits[[best_k]], assign = FALSE)  # samples x K posterior
  ct   <- apply(post, 1, which.max)
  meta <- dat$meta
  meta$CommunityType <- factor(paste0("CT", ct), levels = paste0("CT", 1:best_k))
  meta$Load    <- rowSums(dat$comm)
  meta$Shannon <- vegan::diversity(comm_g, index = "shannon")

  write_tsv(data.frame(sample_id = meta$sample_id, dataset = ds,
                       compartment = as.character(meta$compartment),
                       site = as.character(meta$site),
                       community_type = as.character(meta$CommunityType),
                       load = meta$Load, shannon = meta$Shannon),
            sprintf("tab_eco_dmm_%s.tsv", ds))
  write_tsv(data.frame(dataset = ds, K = k_range, BIC = bic_v, LP = lp_v),
            sprintf("tab_eco_dmm_bic_%s.tsv", ds))

  bic_df <- data.frame(K = k_range, BIC = bic_v, LP = lp_v)
  p_bic <- ggplot(bic_df, aes(K, BIC)) + geom_line(linewidth = 0.8) +
    geom_point(size = 2.2) + geom_vline(xintercept = best_k, linetype = 2, colour = "grey40") +
    scale_x_continuous(breaks = k_range) +
    labs(x = "Number of community types (K)", y = "BIC") + theme_pub()
  p_lp <- ggplot(bic_df, aes(K, LP)) + geom_line(linewidth = 0.8, colour = "#E64B35") +
    geom_point(size = 2.2, colour = "#E64B35") +
    geom_vline(xintercept = best_k, linetype = 2, colour = "grey40") +
    scale_x_continuous(breaks = k_range) +
    labs(x = "Number of community types (K)", y = "Log-likelihood") + theme_pub()

  p_load <- ggplot(meta, aes(CommunityType, Load, fill = CommunityType)) +
    geom_boxplot(outlier.size = 0.6) +
    scale_y_log10(labels = scales::label_scientific()) +
    scale_fill_brewer(palette = "Set2") +
    labs(x = NULL, y = "Absolute load (copies/g)") +
    theme_pub() + theme(legend.position = "none")
  p_shan <- ggplot(meta, aes(CommunityType, Shannon, fill = CommunityType)) +
    geom_boxplot(outlier.size = 0.6) +
    scale_fill_brewer(palette = "Set2") +
    labs(x = "Community type", y = "Shannon index") +
    theme_pub() + theme(legend.position = "none")

  ct_dist <- meta %>% count(compartment, site, CommunityType) %>%
    complete(compartment, site, CommunityType, fill = list(n = 0))
  p_heat <- ggplot(ct_dist, aes(compartment, CommunityType)) +
    geom_tile(aes(fill = n), colour = "white") +
    geom_text(aes(label = n), size = 2.8) +
    facet_grid(~ site) +
    scale_fill_gradient(low = "white", high = "#2166AC") +
    scale_x_discrete(limits = COMP_ORDER) +
    labs(x = "Compartment", y = "Community type", fill = "Count") +
    theme_pub()

  p <- (p_bic | p_lp) / (p_load | p_shan) / p_heat +
    plot_annotation(title = sprintf("%s – DMM community types (absolute abundance)", ds),
                    tag_levels = "A",
                    theme = theme(plot.title = element_text(face = "bold", family = "Times New Roman")))
  ggsave(file.path(FIGS, sprintf("fig_eco_%s_A_dmm.png", ds)), p,
         width = 8.5, height = 9, dpi = 300)
  message(sprintf("  -> figs/fig_eco_%s_A_dmm.png", ds))
  invisible(best_k)
}

# ============================================================================
# ============================================================================
feast_one <- function(source_vec, sink_vec, src_name, sink_name) {
  common <- intersect(names(source_vec), names(sink_vec))
  src_v <- source_vec[common]; sink_v <- sink_vec[common]

  C_mat <- rbind(round(src_v), round(sink_v))
  rownames(C_mat) <- c(src_name, sink_name)
  colnames(C_mat) <- common

  pair_id <- "pair1"
  meta_input <- data.frame(
    SourceSink = c("source", "sink"),
    Env        = c(src_name, sink_name),
    id         = c(pair_id, pair_id),
    row.names  = c(src_name, sink_name)
  )

  tmp_dir <- tempfile(); dir.create(tmp_dir, recursive = TRUE)
  wd_orig <- getwd()
  res <- tryCatch(
    suppressWarnings(FEAST::FEAST(C = C_mat, metadata = meta_input,
                                   EM_iterations = 1000, COVERAGE = NULL,
                                   different_sources_flag = 1,
                                   dir_path = tmp_dir, outfile = "feast_tmp")),
    error = function(e) { message("  FEAST err: ", conditionMessage(e)); NULL },
    finally = setwd(wd_orig)
  )
  if (is.null(res)) return(NA_real_)

  out_file <- file.path(tmp_dir, "feast_tmp_source_contributions_matrix.txt")
  if (file.exists(out_file)) {
    out <- tryCatch(read.delim(out_file, check.names = FALSE),
                    error = function(e) NULL)
    if (!is.null(out)) {
      src_cols <- setdiff(names(out), "Unknown")
      if (length(src_cols) > 0) return(as.numeric(out[1, src_cols[1]]))
    }
  }
  if (is.list(res)) {
    nm <- names(res)
    for (i in seq_along(res)) {
      val <- res[[i]]
      if (is.numeric(val) && !is.null(nm) && nm[i] != "Unknown"
          && !grepl("^data_", nm[i]))
        return(as.numeric(val[1]))
    }
  }
  NA_real_
}

run_feast <- function(ds, dat) {
  comm <- dat$comm; meta <- dat$meta
  pairs <- list(c("BS", "RS"), c("RS", "R"), c("R", "S"), c("S", "L"), c("L", "F"))

  rows <- list()
  for (pair in pairs) {
    src_comp <- pair[1]; sink_comp <- pair[2]
    for (s in SITE_ORDER) {
      src_idx  <- which(meta$site == s & meta$compartment == src_comp)
      sink_idx <- which(meta$site == s & meta$compartment == sink_comp)
      if (length(src_idx) == 0 || length(sink_idx) == 0) next
      composite_src <- colMeans(comm[src_idx, , drop = FALSE])
      for (j in sink_idx) {
        contrib <- feast_one(composite_src, comm[j, ],
                             paste0("Src_", src_comp), meta$sample_id[j])
        rows[[length(rows) + 1]] <- data.frame(
          dataset = ds, site = s, from = src_comp, to = sink_comp,
          sink_sample = meta$sample_id[j], contribution = contrib,
          stringsAsFactors = FALSE)
      }
    }
    message(sprintf("[%s] FEAST cascade %s -> %s done", ds, src_comp, sink_comp))
  }
  cascade <- do.call(rbind, rows)
  write_tsv(cascade, sprintf("tab_eco_feast_%s.tsv", ds))

  summ <- cascade %>% filter(!is.na(contribution)) %>%
    mutate(step = paste(from, to, sep = " -> ")) %>%
    group_by(step, from, to) %>%
    summarise(median = median(contribution), mean = mean(contribution),
              n = n(), .groups = "drop")
  summ$step <- factor(summ$step,
                      levels = paste(COMP_ORDER[-6], COMP_ORDER[-1], sep = " -> "))

  cas_df <- cascade %>% filter(!is.na(contribution)) %>%
    mutate(step = factor(paste(from, to, sep = " -> "), levels = levels(summ$step)),
           site = factor(site, levels = SITE_ORDER))

  p_box <- ggplot(cas_df, aes(step, contribution, fill = from)) +
    geom_boxplot(width = 0.5, outlier.size = 0, alpha = 0.6) +
    geom_jitter(aes(colour = site), width = 0.15, size = 1.2, alpha = 0.7) +
    facet_grid(~ site) +
    scale_fill_manual(values = COMP_COLORS) +
    scale_colour_manual(values = SITE_COLORS) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(x = NULL, y = "Upstream compartment contribution", fill = "Source", colour = "Site") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1),
                        strip.background = element_rect(fill = "grey90"))

  summ_site <- cas_df %>%
    group_by(step, from, to, site) %>%
    summarise(median = median(contribution), n = n(), .groups = "drop")
  allu_long <- summ_site %>%
    mutate(unknown = 1 - median) %>%
    tidyr::pivot_longer(c(median, unknown), names_to = "type", values_to = "value") %>%
    mutate(label = ifelse(type == "median", as.character(from), "Unknown"),
           label = factor(label, levels = c(COMP_ORDER, "Unknown")))
  p_allu <- ggplot(allu_long, aes(x = step, y = value, fill = label)) +
    geom_bar(stat = "identity", width = 0.6) +
    facet_grid(~ site) +
    scale_fill_manual(values = c(COMP_COLORS, Unknown = "grey80")) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    labs(x = NULL, y = "Contribution", fill = "Source") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1),
                        strip.background = element_rect(fill = "grey90"))

  p <- p_box / p_allu +
    plot_annotation(title = sprintf("%s - FEAST source tracking (absolute abundance)", ds),
                    tag_levels = "A",
                    theme = theme(plot.title = element_text(face = "bold", family = "Times New Roman")))
  ggsave(file.path(FIGS, sprintf("fig_eco_%s_B_feast.png", ds)), p,
         width = 12, height = 8, dpi = 300)
  message(sprintf("  -> figs/fig_eco_%s_B_feast.png", ds))
  invisible(summ)
}

# ============================================================================

# Run only the manuscript-reported community typing and FEAST source attribution.
# The 999-randomization iCAMP workflow is in run_icamp_999_randomizations.R.
dir.create(FIGS, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
for (ds in c("16S", "ITS")) {
  message("\nDataset: ", ds)
  dat <- load_dataset(ds)
  run_segment(paste(ds, "DMM"), run_dmm(ds, dat))
  run_segment(paste(ds, "FEAST"), run_feast(ds, dat))
}
message("DMM and FEAST analyses complete.")
