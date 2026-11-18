# ============================================================================
# run_icamp_999_randomizations.R
#
#
#
#
#
# bin.size.limit；ds = 0.2、phylo.metric = "bNTI"、sig.index = "Confidence"、
# ============================================================================

suppressPackageStartupMessages({
  library(ape)
  library(iCAMP)
})
set.seed(20260811)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: Rscript run_icamp_999_randomizations.R <ds> <comp> <rand> <bin.size.limit> <nworker> [out_subdir]")
}
ds   <- args[1]
comp <- args[2]
RAND <- as.integer(args[3])
BSL  <- as.integer(args[4])
NW   <- as.integer(args[5])
OUTSUB <- if (length(args) >= 6 && nzchar(args[6])) args[6] else sprintf("icamp_%s", ds)
stopifnot(ds %in% c("16S", "ITS"), !is.na(RAND), !is.na(BSL), !is.na(NW))
if (!(RAND %in% c(0L, 999L)) || BSL != 12L) {
  stop("Expected rand=999 (or rand=0 for dry-run) and bin.size.limit=12 for the final analysis")
}

ROOT     <- Sys.getenv("KIWI_ROOT", unset = normalizePath(".", winslash = "/"))
META_F   <- file.path(ROOT, "result/r_in/sample_meta.tsv")
OTU_DIR  <- file.path(ROOT, "pre", ds,
                      "Analysis/G1-Compare_All/Absolute_Quantitation/OTU")
COMP_ORDER <- c("BS", "RS", "R", "S", "L", "F")
TAXA_DROP  <- c("#OTU ID", "OTUsize", "Taxonomy", "superkingdom", "phylum",
                "class", "order", "family", "genus", "species")
stopifnot(comp %in% COMP_ORDER)

t_start <- proc.time()
message(sprintf("===== [%s %s] rand=%d bin.size.limit=%d nworker=%d =====",
                ds, comp, RAND, BSL, NW))

# --------------------------------------------------------------- data load --
raw <- read.delim(file.path(OTU_DIR, "otu_copies.txt"), check.names = FALSE,
                  stringsAsFactors = FALSE)
tax_cols  <- intersect(TAXA_DROP, names(raw))
samp_cols <- setdiff(names(raw), tax_cols)

meta <- read.delim(META_F, stringsAsFactors = FALSE)
meta$region <- as.character(meta$region)
meta$region[meta$region == "BZ"] <- "SH"
meta$compartment <- factor(meta$compartment, levels = COMP_ORDER)

keep <- intersect(samp_cols, meta$sample_id)
comm <- as.matrix(raw[keep])
rownames(comm) <- raw[[1]]
storage.mode(comm) <- "double"
comm <- round(comm)
comm <- comm[rowSums(comm) > 0, , drop = FALSE]
comm <- t(comm)                                   # samples x taxa
meta <- meta[match(rownames(comm), meta$sample_id), , drop = FALSE]
rownames(meta) <- meta$sample_id

tree <- read.tree(file.path(OTU_DIR, "rep.seq.tre"))
tips_keep <- tree$tip.label %in% colnames(comm)
tree <- drop.tip(tree, tree$tip.label[!tips_keep])
com  <- intersect(colnames(comm), tree$tip.label)
comm <- comm[, com, drop = FALSE]

prev <- colSums(comm > 0) / nrow(comm)
comm <- comm[, prev >= 0.10, drop = FALSE]
tree <- keep.tip(tree, intersect(tree$tip.label, colnames(comm)))
message(sprintf("  loaded: samples=%d taxa=%d tree_tips=%d",
                nrow(comm), ncol(comm), length(tree$tip.label)))

# --------------------------------------------------- compartment subsetting --
idx <- which(meta$compartment == comp)
stopifnot(length(idx) >= 4)
comm_sub <- comm[idx, , drop = FALSE]
comm_sub <- comm_sub[, colSums(comm_sub) > 0, drop = FALSE]
tree_sub <- keep.tip(tree, colnames(comm_sub))
n_samp <- nrow(comm_sub)
n_taxa <- ncol(comm_sub)
message(sprintf("  [%s] %s: n=%d samples, %d taxa", ds, comp, n_samp, n_taxa))

tmp_dir <- file.path(ROOT, "result/r_out", sprintf("icamp_%s", ds))
out_dir <- file.path(ROOT, "result/r_out", OUTSUB)
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
pd_wd <- file.path(tmp_dir, sprintf("pd_%s_%s", ds, comp))
spn   <- sprintf("pdsn_%s.txt", comp)
des   <- sprintf("pddc_%s.desc", comp)
bkn   <- sprintf("pdbk_%s.bin", comp)

# ------------------------------- phylogenetic distance (reuse when valid) ----
reuse <- FALSE
spname_vec <- NULL
if (file.exists(file.path(pd_wd, des)) && file.exists(file.path(pd_wd, spn))) {
  spname_vec <- tryCatch({
    d <- read.csv(file.path(pd_wd, spn), check.names = FALSE,
                  stringsAsFactors = FALSE)
    if ("taxon.name" %in% names(d)) as.character(d[["taxon.name"]]) else character(0)
  }, error = function(e) character(0))
  reuse <- length(spname_vec) == n_taxa &&
           setequal(spname_vec, as.character(colnames(comm_sub)))
  message(sprintf("  pdist check: %d names on file vs %d taxa -> reuse=%s",
                  length(spname_vec), n_taxa, reuse))
}

# ------------------------------------------------------------ dry run gate ---
if (RAND == 0L) {
  message(sprintf("  DRY RUN [%s %s] n_samples=%d n_taxa=%d pdist_reuse=%s",
                  ds, comp, n_samp, n_taxa, reuse))
  quit(status = 0)
}

if (reuse) {
  message("  reusing existing pdist (taxa set identical)")
  pd <- list(pd.desc = des, pd.spname = spname_vec, pd.wd = pd_wd)
} else {
  message("  computing pdist.big ...")
  if (dir.exists(pd_wd)) unlink(pd_wd, recursive = TRUE)
  dir.create(pd_wd, recursive = TRUE)
  pd <- iCAMP::pdist.big(tree = tree_sub, wd = pd_wd, nworker = NW,
                         pd.spname.file = spn, pd.backingfile = bkn,
                         pd.desc.file = des)
}

# ------------------------------------------------------------- icamp.big ----
gc()
res <- iCAMP::icamp.big(comm = comm_sub, tree = tree_sub,
                        pd.desc = pd$pd.desc, pd.spname = pd$pd.spname,
                        pd.wd = pd_wd, rand = RAND,
                        prefix = paste0(ds, "_", comp),
                        ds = 0.2, bin.size.limit = BSL, nworker = NW,
                        output.wd = out_dir, phylo.metric = "bNTI",
                        sig.index = "Confidence")

elapsed <- (proc.time() - t_start)[["elapsed"]]
out_csv <- file.path(out_dir,
                     sprintf("%s_%s.process.CbMNTDiCBraya.csv", ds, comp))
stopifnot(file.exists(out_csv))
cbm <- read.csv(out_csv, check.names = FALSE)

# --------------------------------------------------- provenance (runinfo) ----
runinfo <- data.frame(
  dataset        = ds,
  compartment    = comp,
  rand           = RAND,
  bin.size.limit = BSL,
  nworker        = NW,
  n_samples      = n_samp,
  n_taxa         = n_taxa,
  n_pairs        = nrow(cbm),
  elapsed_sec    = round(elapsed, 1),
  pdist_reused   = reuse,
  seed           = 20260811,
  downsampling   = 0.2,
  phylo_metric   = "bNTI",
  sig_index      = "Confidence",
  R_version      = paste0(R.version$major, ".", R.version$minor),
  iCAMP_version  = as.character(packageVersion("iCAMP")),
  script         = "run_icamp_999_randomizations.R",
  timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  res_elements   = paste(names(res), collapse = "|"),
  stringsAsFactors = FALSE)
write.table(runinfo,
            file.path(out_dir, sprintf("_runinfo_%s_%s.tsv", ds, comp)),
            sep = "\t", quote = FALSE, row.names = FALSE)

message(sprintf("  DONE [%s %s] pairs=%d elapsed=%.1f min -> %s",
                ds, comp, nrow(cbm), elapsed / 60, basename(out_csv)))
