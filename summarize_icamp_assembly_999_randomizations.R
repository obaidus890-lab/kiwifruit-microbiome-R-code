# ============================================================================
# summarize_icamp_assembly_999_randomizations.R
#
#
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(vegan)
})

ROOT <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
stopifnot(dir.exists(ROOT), dir.exists(file.path(ROOT, "result/tables")))
TABLES <- file.path(ROOT, "result/tables")
META_F <- file.path(ROOT, "result/r_in/sample_meta.tsv")

cmd <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath(".")
source_script <- file.path(script_dir, "analyze_icamp_environment_and_spatial_patterns.R")
if (!file.exists(source_script)) stop("Required sibling script not found: ", source_script)
src <- readLines(source_script, warn = FALSE)
cut <- grep("^# main$", src)
stopifnot(length(cut) >= 1)
eval(parse(text = paste(src[1:(cut[1] - 2)], collapse = "\n")), envir = globalenv())
ggsave <- function(...) invisible(NULL)
message("[ok] analyze_icamp_environment_and_spatial_patterns.R functions loaded (up to line ", cut[1] - 2, ")")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
revise_icamp_tables <- function(ds) {
  icamp_dir <- file.path(ROOT, "result/r_out", sprintf("icamp_%s", ds))
  csv_files <- list.files(icamp_dir,
                          pattern = sprintf("^%s_.*\\.process\\.CbMNTDiCBraya\\.csv$", ds),
                          full.names = TRUE)
  stopifnot(length(csv_files) > 0)

  meta <- read.delim(META_F, stringsAsFactors = FALSE)
  meta$region <- as.character(meta$region)
  meta$region[meta$region == "BZ"] <- "SH"
  sid2site <- setNames(meta$region, meta$sample_id)

  proc_map <- c(
    "Heterogeneous.Selection" = "Variable selection",
    "Homogeneous.Selection"   = "Homogeneous selection",
    "Dispersal.Limitation"    = "Dispersal limitation",
    "Homogenizing.Dispersal"  = "Homogenizing dispersal",
    "Drift.and.Others"        = "Drift and others")
  proc_order <- c("Variable selection", "Homogeneous selection",
                  "Dispersal limitation", "Homogenizing dispersal",
                  "Drift and others")
  det_procs <- c("Variable selection", "Homogeneous selection")

  qg_all <- list()
  for (csv_file in csv_files) {
    comp <- sub(sprintf("^%s_(.+?)\\.process.*", ds), "\\1", basename(csv_file))
    cbm <- read.csv(csv_file, check.names = FALSE)
    proc_cols <- setdiff(names(cbm), c("", "sample1", "sample2"))
    s1_site <- sid2site[cbm$sample1]
    s2_site <- sid2site[cbm$sample2]
    within <- !is.na(s1_site) & !is.na(s2_site) & s1_site == s2_site
    cbm_w <- cbm[within, , drop = FALSE]
    if (nrow(cbm_w) == 0) next
    cbm_w$site <- s1_site[within]
    for (st in unique(cbm_w$site)) {
      sub <- cbm_w[cbm_w$site == st, , drop = FALSE]
      qp_vec <- colMeans(sub[, proc_cols], na.rm = TRUE)
      qg_all[[length(qg_all) + 1]] <- data.frame(
        Compartment = comp, Site = st, Process = names(qp_vec),
        Proportion = as.numeric(qp_vec), dataset = ds,
        stringsAsFactors = FALSE)
    }
  }
  stopifnot(length(qg_all) > 0)
  qg_df <- do.call(rbind, qg_all)
  qg_df$Compartment <- factor(qg_df$Compartment, levels = COMP_ORDER)
  qg_df$Site <- factor(qg_df$Site, levels = SITE_ORDER)
  qg_df$Process <- factor(proc_map[qg_df$Process], levels = proc_order)
  qg_df$Category <- ifelse(qg_df$Process %in% det_procs,
                           "Deterministic", "Stochastic")
  write.table(qg_df, file.path(TABLES, sprintf("tab_eco_icamp_%s.tsv", ds)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("  -> tables/tab_eco_icamp_%s.tsv (%d rows)", ds, nrow(qg_df)))
  invisible(qg_df)
}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
spatial_tables <- function(ds) {
  icamp_dir <- file.path(ROOT, "result/r_out", sprintf("icamp_%s", ds))
  csv_files <- sort(list.files(icamp_dir,
                               pattern = sprintf("^%s_(.+?)\\.process\\.CbMNTDiCBraya\\.csv$", ds),
                               full.names = TRUE))
  pp <- list()
  for (cf in csv_files) {
    comp <- sub(sprintf("^%s_(.+)\\.process.*", ds), "\\1", basename(cf))
    cbm <- read.csv(cf, check.names = FALSE)
    pm <- proc_matrix(cbm)
    keep <- sample_site(cbm$sample1) == sample_site(cbm$sample2)
    if (!any(keep)) next
    d <- data.frame(pair = seq_len(sum(keep)),
                    site = sample_site(cbm$sample1)[keep],
                    compartment = comp, pm[keep, , drop = FALSE],
                    check.names = FALSE)
    d$det <- rowSums(d[, DET_PROCS])
    pp[[length(pp) + 1]] <- d
  }
  pp <- do.call(rbind, pp)
  pp$compartment <- factor(pp$compartment, levels = COMP_ORDER)
  pp$site <- factor(pp$site, levels = SITE_ORDER)

  BG <- c("BS", "RS", "R"); AG <- c("S", "L", "F")
  wg <- do.call(rbind, lapply(SITE_ORDER, function(st) {
    sub <- pp[pp$site == st, ]
    wt <- wilcox.test(sub$det[sub$compartment %in% BG],
                      sub$det[sub$compartment %in% AG])
    data.frame(site = st, n_bg = sum(sub$compartment %in% BG),
               n_ag = sum(sub$compartment %in% AG),
               median_bg = median(sub$det[sub$compartment %in% BG]),
               median_ag = median(sub$det[sub$compartment %in% AG]),
               wilcox_p = wt$p.value)
  }))
  write.table(wg, file.path(TABLES, sprintf("tab_eco_icamp_spatial_%s.tsv", ds)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("  -> tables/tab_eco_icamp_spatial_%s.tsv", ds))
  invisible(wg)
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
for (ds in c("16S", "ITS")) {
  message(sprintf("\n===== %s: rebuild eco tables (rand=999 outputs) =====", ds))
  revise_icamp_tables(ds)
  bin_all <- icamp_bin_process(ds)
  icamp_env_corr(ds)
  spatial_tables(ds)
}
message("\nAll eco tables rebuilt.")
