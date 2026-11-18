# ============================================================================
# analyze_icamp_environment_and_spatial_patterns.R
# Extended iCAMP analyses per dataset (16S / ITS):
#   D. Bin-level heatmap of assembly processes (from detail .rda)
#   E. Environmental correlates: Spearman / Mantel / VPA (climate vs soil)
#   F. Spatial dynamics of deterministic fraction along the soil-plant
#      continuum (single time point study -> space only)
# Outputs: fig_eco_{ds}_D_icamp_bin_heatmap.png
#          fig_eco_{ds}_E_icamp_env.png
#          fig_eco_{ds}_F_icamp_spatial.png
#          result/tables/tab_eco_icamp_bin_process_{ds}.tsv
#          result/tables/tab_eco_icamp_env_corr_{ds}.tsv
#          result/tables/tab_eco_icamp_vpa_{ds}.tsv
# ============================================================================
set.seed(20260812)

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(patchwork)
  library(scales); library(vegan)
})

ROOT   <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
FIGS   <- file.path(ROOT, "result/figs")
TABLES <- file.path(ROOT, "result/tables")
RIN    <- file.path(ROOT, "result/r_in")
META_F <- file.path(RIN, "sample_meta.tsv")

COMP_ORDER  <- c("BS", "RS", "R", "S", "L", "F")
COMP_LABEL  <- c(BS = "Bulk soil", RS = "Rhizosphere soil", R = "Root",
                 S = "Stem", L = "Leaf", F = "Fruit")
SITE_ORDER  <- c("WH", "SH", "SX")
SITE_COLORS <- c(WH = "#3C5488", SH = "#E64B35", SX = "#00A087")

PROC_MAP <- c(
  "Heterogeneous.Selection" = "Variable selection",
  "Homogeneous.Selection"   = "Homogeneous selection",
  "Dispersal.Limitation"    = "Dispersal limitation",
  "Homogenizing.Dispersal"  = "Homogenizing dispersal",
  "Drift.and.Others"        = "Drift and others")
PROC_ORDER <- c("Variable selection", "Homogeneous selection",
                "Dispersal limitation", "Homogenizing dispersal",
                "Drift and others")
PROC_COLORS <- c("Variable selection"     = "#D7191C",
                 "Homogeneous selection"  = "#FDAE61",
                 "Dispersal limitation"   = "#ABD9E9",
                 "Homogenizing dispersal" = "#2C7BB6",
                 "Drift and others"       = "#8C8C8C")
DET_PROCS <- c("Variable selection", "Homogeneous selection")
ENV_VARS  <- c("pH", "SOM", "TN", "HN", "NH4-N", "NO2-N", "TP", "AP", "TK",
               "AK", "Urease", "Invertase", "Catalase", "Cellulase",
               "Phosphatase", "mean_temp", "rel_humidity", "total_precip")
# environmental variable blocks for the grouped heatmap (panel E)
ENV_GROUP <- c(setNames(rep("Soil chemistry", 10), ENV_VARS[1:10]),
               setNames(rep("Enzyme activity", 5), ENV_VARS[11:15]),
               setNames(rep("Climate", 3), ENV_VARS[16:18]))

FIGURE_FONT_FAMILY <- "Times New Roman"

theme_pub <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = FIGURE_FONT_FAMILY) +
    theme(panel.border = element_rect(fill = NA, colour = "black", linewidth = 0.8),
          axis.text = element_text(colour = "black"),
          legend.key = element_rect(fill = NA))
}

# sample prefix -> site (BZ is the SH site)
sample_site <- function(ids) {
  pfx <- sub("_.*", "", ids)
  ifelse(pfx == "BZ", "SH", pfx)
}

# per-pair process matrix (columns = PROC_ORDER) from the CbMNTDiCBraya table
proc_matrix <- function(cbm) {
  raw <- setdiff(names(cbm), c("", "sample1", "sample2"))
  m <- as.matrix(cbm[, raw, drop = FALSE])
  colnames(m) <- PROC_MAP[raw]
  m
}

# ----------------------------------------------------------------------------
# 1. Bin-level process proportions (iCAMP bin classification per pair)
# ----------------------------------------------------------------------------
icamp_bin_process <- function(ds) {
  icamp_dir <- file.path(ROOT, "result/r_out", sprintf("icamp_%s", ds))
  rda_files <- list.files(icamp_dir, pattern = "\\.iCAMP\\.Confidence\\.detail\\.rda$",
                          full.names = TRUE)
  stopifnot(length(rda_files) > 0)

  otu <- read.delim(file.path(ROOT, ds, "otu.xls"), check.names = FALSE,
                    stringsAsFactors = FALSE)
  asv_tax <- data.frame(asv = otu[[1]], genus = otu[["genus"]],
                        phylum = otu[["phylum"]], stringsAsFactors = FALSE)
  asv_tax$genus[asv_tax$genus %in% c("", "unclassified")] <- NA

  out <- list()
  for (rda_f in rda_files) {
    comp <- sub(sprintf("^%s_(.+)\\.iCAMP.*", ds), "\\1", basename(rda_f))
    res <- NULL
    load(rda_f)
    det <- res$detail

    bin_ids <- as.integer(sub("^CbMNTDi\\.bin(\\d+)$", "\\1",
                              grep("^CbMNTDi\\.bin", names(det$SigbMNTDi), value = TRUE)))
    stopifnot(identical(paste(det$SigbMNTDi$name1, det$SigbMNTDi$name2),
                        paste(det$SigBCa$name1, det$SigBCa$name2)),
              identical(paste(det$SigbMNTDi$name1, det$SigbMNTDi$name2),
                        paste(det$bin.weight$samp1, det$bin.weight$samp2)))
    bnti <- as.matrix(det$SigbMNTDi[, paste0("CbMNTDi.bin", bin_ids)])
    rc   <- as.matrix(det$SigBCa[, paste0("Cbraya.bin", bin_ids)])
    wt   <- as.matrix(det$bin.weight[, paste0("bin", bin_ids)])
    stopifnot(nrow(bnti) == nrow(rc), nrow(rc) == nrow(wt))

    # iCAMP bin classification: bNTI cut +/-2, Raup-Crick cut +/-0.95
    proc_bin <- matrix("Drift and others", nrow(bnti), length(bin_ids))
    proc_bin[bnti < -2] <- "Homogeneous selection"
    proc_bin[bnti >  2] <- "Variable selection"
    mid <- bnti >= -2 & bnti <= 2
    proc_bin[mid & rc >  0.95] <- "Dispersal limitation"
    proc_bin[mid & rc < -0.95] <- "Homogenizing dispersal"

    wp <- lapply(seq_along(bin_ids), function(j) {
      s <- tapply(wt[, j], proc_bin[, j], sum)
      s <- s[PROC_ORDER]
      s[is.na(s)] <- 0
      names(s) <- PROC_ORDER
      s / sum(wt[, j])
    })
    names(wp) <- bin_ids
    bin_df <- do.call(rbind, lapply(bin_ids, function(b) {
      w <- wp[[as.character(b)]]
      data.frame(bin = b, Process = names(w), prop = as.numeric(w))
    }))
    mean_w <- vapply(bin_ids, function(b) mean(wt[, b]), numeric(1))

    # bin label = taxonomy of its most abundant member ASV
    comm <- res$detail$comm
    if (!is.numeric(comm)) comm <- apply(comm, 2, as.numeric)
    tot_abund <- colSums(comm)
    names(tot_abund) <- colnames(comm)
    binsp <- det$taxabin$bin.united.sp
    lab <- vapply(bin_ids, function(b) {
      sp <- binsp[[b]]
      ab <- tot_abund[sp]; ab <- ab[!is.na(ab)]
      if (length(ab) == 0) return("unidentified")
      top <- names(ab)[which.max(ab)]
      g <- asv_tax$genus[match(top, asv_tax$asv)]
      if (!is.na(g)) return(g)
      ph <- asv_tax$phylum[match(top, asv_tax$asv)]
      if (is.na(ph) || ph == "") return("unidentified")
      paste0(ph, "*")
    }, character(1))
    lab <- make.unique(lab)

    bin_df$label <- lab[match(bin_df$bin, bin_ids)]
    bin_df$mean_weight <- mean_w[match(bin_df$bin, bin_ids)]
    bin_df$Compartment <- comp
    out[[length(out) + 1]] <- bin_df
  }

  bin_all <- do.call(rbind, out)
  bin_all$Process <- factor(bin_all$Process, levels = PROC_ORDER)
  bin_all$Compartment <- factor(bin_all$Compartment, levels = COMP_ORDER)
  bin_all$dataset <- ds
  write.table(bin_all, file.path(TABLES, sprintf("tab_eco_icamp_bin_process_%s.tsv", ds)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  bin_all
}

# ----------------------------------------------------------------------------
# 2. Bin-level heatmap figure
# ----------------------------------------------------------------------------
plot_bin_heatmap <- function(ds, bin_all) {
  # bin numbers are compartment-local -> pick the 8 most abundant bins
  # (by mean weight) WITHIN each compartment; free_y so every panel
  # shows only its own bins
  top <- bin_all %>%
    group_by(Compartment, bin) %>%
    summarise(mw = mean(mean_weight), .groups = "drop") %>%
    group_by(Compartment) %>%
    arrange(desc(mw), .by_group = TRUE) %>%
    slice_head(n = 8) %>%
    mutate(cb = paste(Compartment, bin))
  bin_all <- bin_all[paste(bin_all$Compartment, bin_all$bin) %in% top$cb, ]

  bo <- bin_all %>%
    group_by(Compartment, bin, label) %>%
    summarise(mean_w = mean(mean_weight),
              det = sum(prop[Process %in% DET_PROCS]), .groups = "drop") %>%
    arrange(Compartment, desc(det), desc(mean_w))
  bo$row <- paste(bo$Compartment, bo$bin, bo$label, sep = "|")

  hm <- bin_all %>%
    group_by(Compartment, Process, bin, label) %>%
    summarise(prop = sum(prop), .groups = "drop") %>%
    mutate(row = paste(Compartment, bin, label, sep = "|"))
  hm$row <- factor(hm$row, levels = bo$row)
  hm$Process <- factor(as.character(hm$Process), levels = PROC_ORDER)

  ggplot(hm, aes(Process, row, fill = prop)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    facet_grid(Compartment ~ ., scales = "free_y", space = "free_y") +
    scale_fill_viridis_c(option = "C", limits = c(0, 1), name = "Process\nproportion") +
    scale_y_discrete(labels = function(z) sub("^[^|]+\\|[0-9]+\\|", "", z)) +
    labs(x = NULL, y = NULL,
         title = sprintf("%s - iCAMP bin-level assembly processes (top 8 bins per compartment)", ds)) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
          axis.text.y = element_text(size = 6),
          strip.text.y = element_text(angle = 0, size = 7),
          strip.background = element_rect(fill = "grey90"),
          plot.title = element_text(face = "bold"))
}

# ----------------------------------------------------------------------------
# 3. Environmental correlates: Spearman / Mantel / VPA
# ----------------------------------------------------------------------------
icamp_env_corr <- function(ds) {
  icamp_dir <- file.path(ROOT, "result/r_out", sprintf("icamp_%s", ds))
  csv_files <- sort(list.files(icamp_dir,
                               pattern = sprintf("^%s_.*\\.process\\.CbMNTDiCBraya\\.csv$", ds),
                               full.names = TRUE))
  stopifnot(length(csv_files) > 0)
  env <- read.delim(file.path(RIN, "env_n18.tsv"), check.names = FALSE,
                    stringsAsFactors = FALSE)
  meta <- read.delim(META_F, stringsAsFactors = FALSE)
  meta$region[meta$region == "BZ"] <- "SH"
  sid2site <- setNames(as.character(meta$region), meta$sample_id)
  reg_env <- aggregate(as.matrix(env[, ENV_VARS]),
                       by = list(region = env$region), mean)
  E_scaled <- scale(reg_env[, ENV_VARS])
  rownames(E_scaled) <- reg_env$region

  cells <- list()
  for (cf in csv_files) {
    comp <- sub(sprintf("^%s_(.+?)\\.process.*", ds), "\\1", basename(cf))
    cbm <- read.csv(cf, check.names = FALSE)
    pm <- proc_matrix(cbm)
    keep <- sample_site(cbm$sample1) == sample_site(cbm$sample2)
    if (!any(keep)) next
    d <- data.frame(pm[keep, , drop = FALSE],
                    region = sample_site(cbm$sample1[keep]),
                    check.names = FALSE) %>%
      group_by(region) %>%
      summarise(across(everything(), mean), .groups = "drop")
    d$compartment <- comp
    d$cell_id <- paste(comp, d$region, sep = "_")
    cells[[length(cells) + 1]] <- d
  }
  cells <- do.call(rbind, cells)
  cells <- merge(cells, env[, c("cell_id", ENV_VARS)], by = "cell_id")
  stopifnot(nrow(cells) == 18)
  rownames(cells) <- cells$cell_id

  metrics <- cells[, PROC_ORDER]
  metrics$Deterministic <- rowSums(metrics[, DET_PROCS])

  # Spearman
  rho <- cor(as.matrix(metrics), as.matrix(cells[, ENV_VARS]),
             method = "spearman", use = "pairwise.complete.obs")
  pmat <- matrix(NA_real_, nrow(rho), ncol(rho),
                 dimnames = dimnames(rho))
  for (m in rownames(rho)) for (v in colnames(rho))
    pmat[m, v] <- cor.test(metrics[[m]], cells[[v]], method = "spearman",
                           exact = FALSE)$p.value
  qmat <- matrix(p.adjust(as.vector(pmat), "BH"), nrow(pmat), ncol(pmat),
                 dimnames = dimnames(pmat))

  # Mantel: process composition vs environment
  dm_proc <- vegdist(metrics, method = "bray")
  dm_env  <- dist(scale(cells[, ENV_VARS]))
  man_g <- mantel(dm_proc, dm_env, permutations = 999)
  man_rows <- data.frame(scope = "all_cells", test = "mantel",
                         r = man_g$statistic, p = man_g$signif)
  # per compartment: Spearman correlation between the deterministic fraction
  # of each sample pair and the environmental distance between its two sites
  man_rows <- rbind(man_rows, do.call(rbind, lapply(COMP_ORDER, function(cp) {
    cf <- csv_files[basename(csv_files) ==
                      sprintf("%s_%s.process.CbMNTDiCBraya.csv", ds, cp)]
    if (length(cf) == 0) return(NULL)
    cbm <- read.csv(cf, check.names = FALSE)
    pm_cp <- proc_matrix(cbm)
    det_pair <- rowSums(pm_cp[, DET_PROCS, drop = FALSE])
    st1 <- sid2site[cbm$sample1]; st2 <- sid2site[cbm$sample2]
    de <- sqrt(rowSums((E_scaled[st1, , drop = FALSE] -
                        E_scaled[st2, , drop = FALSE])^2))
    ct <- cor.test(det_pair, de, method = "spearman", exact = FALSE)
    data.frame(scope = cp, test = "spearman_pair",
               r = unname(ct$estimate), p = ct$p.value)
  })))
  mantel_label <- sprintf(paste0("Mantel (process vs environment): r = %.3f, p = %.3f; ",
                                 "*P < 0.05, \u2020P < 0.10 (uncorrected; BH q in Table S13)"),
                          man_g$statistic, man_g$signif)

  # VPA: climate vs soil (both summarized by regional PCA axes; only 3 regions,
  # raw climate variables are rank 2 -> collinear)
  climate_vars <- c("mean_temp", "rel_humidity", "total_precip")
  soil_vars <- ENV_VARS[!(ENV_VARS %in% climate_vars)]
  reg <- aggregate(as.matrix(env[, ENV_VARS]),
                   by = list(region = env$region), mean)
  pc_cli <- prcomp(scale(reg[, climate_vars]), center = FALSE, scale. = FALSE)
  pc_soil <- prcomp(scale(reg[, soil_vars]), center = FALSE, scale. = FALSE)
  pcsc <- data.frame(region = reg$region,
                     cliPC1 = pc_cli$x[, 1],
                     soilPC1 = pc_soil$x[, 1], soilPC2 = pc_soil$x[, 2])
  cells_vpa <- merge(cells, pcsc, by = "region")
  Y <- decostand(as.matrix(metrics), "hellinger")
  vp <- varpart(Y, cells_vpa[, "cliPC1", drop = FALSE],
                cells_vpa[, c("soilPC1", "soilPC2")])
  ind <- vp$part$indfract
  adj <- ind$Adj.R.squared
  rn <- rownames(ind)
  a  <- adj[grepl("^\\[a\\]", rn)]   # pure climate
  b  <- adj[grepl("^\\[b\\]", rn)]   # pure soil
  cc <- adj[grepl("^\\[c\\]", rn)]   # shared
  fr <- data.frame(
    fraction = c("Pure climate", "Shared", "Pure soil", "Unexplained"),
    value = c(max(0, a), max(0, cc), max(0, b),
              max(0, 1 - (max(0, a) + max(0, cc) + max(0, b)))),
    adj_R2_raw = c(a, cc, b, 1 - (a + cc + b)))
  ann_p <- tryCatch({
    a1 <- anova(rda(Y ~ cliPC1, data = cells_vpa), permutations = 999)
    a2 <- anova(rda(Y ~ soilPC1 + soilPC2, data = cells_vpa), permutations = 999)
    c(climate = a1$`Pr(>F)`[1], soil = a2$`Pr(>F)`[1])
  }, error = function(e) c(climate = NA_real_, soil = NA_real_))
  star_p <- function(p) ifelse(is.na(p) || p >= 0.05, "", "*")
  vpa_note <- sprintf("RDA permutation: climate p = %.3f%s, soil p = %.3f%s",
                      ann_p[1], star_p(ann_p[1]), ann_p[2], star_p(ann_p[2]))

  # figures
  long <- data.frame(metric = rep(rownames(rho), ncol(rho)),
                     env = rep(colnames(rho), each = nrow(rho)),
                     rho = as.vector(rho), p = as.vector(pmat),
                     q = as.vector(qmat))
  long$env <- factor(long$env, levels = ENV_VARS)
  long$grp <- factor(ENV_GROUP[as.character(long$env)],
                     levels = c("Soil chemistry", "Enzyme activity", "Climate"))
  long$metric <- factor(long$metric, levels = rev(c(PROC_ORDER, "Deterministic")))
  # uncorrected-P markers (BH q values are reported in Table S13)
  long$star <- ifelse(long$p < 0.05, "*", ifelse(long$p < 0.1, "\u2020", ""))

  p_sp <- ggplot(long, aes(env, metric, fill = rho)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = star), size = 2.8) +
    facet_grid(~ grp, scales = "free_x", space = "free_x") +
    scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C",
                         midpoint = 0, limits = c(-1, 1),
                         name = expression(rho)) +
    labs(x = NULL, y = NULL) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6),
          axis.text.y = element_text(size = 6),
          strip.background = element_rect(fill = "grey92"),
          strip.text.x = element_text(size = 6.5, face = "bold"),
          panel.spacing = unit(2, "pt"))

  fp <- fr[fr$fraction != "Unexplained", ]
  fp$fraction <- factor(fp$fraction,
                        levels = c("Pure climate", "Pure soil", "Shared"))
  p_vpa <- ggplot(fp, aes(y = fraction, x = value, fill = fraction)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.1f%%", value * 100)),
              hjust = -0.2, size = 2.6) +
    scale_fill_manual(values = c("Pure climate" = "#3C5488",
                                 "Pure soil" = "#E64B35",
                                 "Shared" = "#8491B4")) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05)),
                       limits = c(0, max(fp$value) * 1.35),
                       labels = percent_format()) +
    labs(x = expression(Adjusted ~ R^2), y = NULL,
         title = "VPA: climate vs soil") +
    theme_pub() +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 9),
          axis.text.y = element_text(size = 6.5))

  vpa_cap <- sprintf("Unexplained: %.1f%%; explained: %.1f%%. %s",
                     fr$value[fr$fraction == "Unexplained"] * 100,
                     (1 - fr$value[fr$fraction == "Unexplained"]) * 100,
                     vpa_note)

  p <- (p_sp | p_vpa) + plot_layout(widths = c(2.5, 1)) +
    plot_annotation(
      title = sprintf("%s - iCAMP processes vs environment (n = 18 cells)", ds),
      subtitle = mantel_label,
      caption = vpa_cap,
      tag_levels = "a",
      theme = theme(plot.title = element_text(face = "bold",
                                              family = FIGURE_FONT_FAMILY),
                    plot.subtitle = element_text(size = 7,
                                                 family = FIGURE_FONT_FAMILY),
                    plot.caption = element_text(size = 6.5, hjust = 0,
                                                family = FIGURE_FONT_FAMILY)))
  ggsave(file.path(FIGS, sprintf("fig_eco_%s_E_icamp_env.png", ds)), p,
         width = 7.2, height = 4.8, dpi = 300)
  message(sprintf("  -> figs/fig_eco_%s_E_icamp_env.png", ds))

  sp_tab <- long
  sp_tab$test <- "spearman"
  mantel_out <- data.frame(metric = "process_composition",
                           env = man_rows$scope,
                           rho = man_rows$r, p = man_rows$p, q = NA_real_,
                           grp = "overall", star = "",
                           test = man_rows$test)
  write.table(rbind(sp_tab[, names(mantel_out)], mantel_out),
              file.path(TABLES, sprintf("tab_eco_icamp_env_corr_%s.tsv", ds)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(fr, file.path(TABLES, sprintf("tab_eco_icamp_vpa_%s.tsv", ds)),
              sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("  [%s] Mantel all cells r=%.3f p=%.3f; VPA adjR2: climate=%.3f shared=%.3f soil=%.3f",
                  ds, man_g$statistic, man_g$signif, a, cc, b))
  invisible(NULL)
}

# ----------------------------------------------------------------------------
# 4. Spatial dynamics along the soil-plant continuum
# ----------------------------------------------------------------------------
plot_spatial <- function(ds) {
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
    d <- data.frame(pair = seq_len(sum(keep)), site = sample_site(cbm$sample1)[keep],
                    compartment = comp, pm[keep, , drop = FALSE],
                    check.names = FALSE)
    d$det <- rowSums(d[, DET_PROCS])
    pp[[length(pp) + 1]] <- d
  }
  pp <- do.call(rbind, pp)
  pp$compartment <- factor(pp$compartment, levels = COMP_ORDER)
  pp$x <- as.integer(pp$compartment)
  pp$site <- factor(pp$site, levels = SITE_ORDER)

  # belowground vs aboveground deterministic fraction (Wilcoxon per site)
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
  fmt_p <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.3f", p))
  site_lab <- setNames(sprintf("%s  (P %s)", wg$site, fmt_p(wg$wilcox_p)),
                       wg$site)

  summ <- pp %>% group_by(site, compartment, x) %>%
    summarise(det_m = mean(det), det_sd = sd(det), .groups = "drop")

  p1 <- ggplot(summ, aes(x, det_m, colour = site)) +
    geom_ribbon(aes(ymin = det_m - det_sd, ymax = det_m + det_sd, fill = site),
                alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    geom_vline(xintercept = 3.5, linetype = 2, colour = "grey50") +
    geom_hline(yintercept = 0.5, linetype = 3, colour = "grey40",
               linewidth = 0.35) +
    annotate("text", x = 1.05, y = 1, label = "Belowground", size = 2.4, hjust = 0,
             colour = "grey40") +
    annotate("text", x = 4.2, y = 1, label = "Aboveground", size = 2.4, hjust = 0,
             colour = "grey40") +
    facet_grid(~ site, labeller = as_labeller(site_lab)) +
    scale_colour_manual(values = SITE_COLORS) +
    scale_fill_manual(values = SITE_COLORS) +
    scale_x_continuous(breaks = 1:6, labels = unname(COMP_LABEL)) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    labs(x = NULL, y = "Deterministic fraction", colour = NULL, fill = NULL) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
          strip.background = element_rect(fill = "grey90"),
          legend.position = "none")

  qg <- pp %>% pivot_longer(all_of(PROC_ORDER), names_to = "Process",
                            values_to = "v") %>%
    group_by(site, compartment, Process) %>%
    summarise(Proportion = mean(v), .groups = "drop") %>%
    mutate(Process = factor(Process, levels = PROC_ORDER))
  p2 <- ggplot(qg, aes(compartment, Proportion, fill = Process)) +
    geom_bar(stat = "identity", width = 0.7) +
    facet_grid(~ site) +
    scale_x_discrete(limits = COMP_ORDER, labels = COMP_LABEL) +
    scale_y_continuous(expand = c(0, 0), labels = percent_format()) +
    scale_fill_manual(values = PROC_COLORS) +
    labs(x = "Compartment", y = "Relative contribution", fill = "Process") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          strip.background = element_rect(fill = "grey90"))

  p <- (p1 / p2) +
    plot_annotation(title = sprintf("%s - spatial dynamics of community assembly along the continuum", ds),
                    tag_levels = "a",
                    theme = theme(plot.title = element_text(face = "bold",
                                                            family = FIGURE_FONT_FAMILY)))
  ggsave(file.path(FIGS, sprintf("fig_eco_%s_F_icamp_spatial.png", ds)), p,
         width = 7.2, height = 5.2, dpi = 300)
  message(sprintf("  -> figs/fig_eco_%s_F_icamp_spatial.png", ds))
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
for (ds in c("16S", "ITS")) {
  message(sprintf("\n===== %s: extended iCAMP =====", ds))
  bin_all <- icamp_bin_process(ds)
  p_bin <- plot_bin_heatmap(ds, bin_all)
  ggsave(file.path(FIGS, sprintf("fig_eco_%s_D_icamp_bin_heatmap.png", ds)),
         p_bin, width = 7.2, height = 7.4, dpi = 300)
  message(sprintf("  -> figs/fig_eco_%s_D_icamp_bin_heatmap.png", ds))
  icamp_env_corr(ds)
  plot_spatial(ds)
}
message("\nDone.")
