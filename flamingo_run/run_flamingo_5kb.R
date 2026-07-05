#!/usr/bin/env Rscript
# Real FLAMINGO at its NATIVE 5 kb resolution on GM12878 chr22 mappable arm
# (16-51.3 Mb, re-based to 0). Same KR contacts, scored by the same dSCC.
# Run from flamingo_run/ (writes Domain_data/ Genomic_loc/ here).

suppressPackageStartupMessages({
  library(FLAMINGOr); library(Matrix); library(data.table); library(mgcv)
  library(progress); library(parallel); library(prodlim)
})

frag_res   <- 5e3
domain_res <- 1e6
contact_file <- "chr22_5kb.contacts.txt"
norm_high <- "ones_high_5kb.txt"
norm_low  <- "ones_low_5kb.txt"
n_bins   <- length(readLines(norm_high))       # kept in sync with the export
chr_size <- n_bins * frag_res
chr_name <- "chr22"

SEED <- as.integer(Sys.getenv("FSEED","0")); set.seed(SEED)
cat("FLAMINGO 5kb: n_bins=", n_bins, " chr_size=", chr_size, " seed=", SEED, "\n")

res <- flamingo.main_func(
  hic_data_low       = contact_file,
  hic_data_high      = contact_file,
  file_format        = "sparse matrix",
  domain_res         = domain_res,
  frag_res           = frag_res,
  chr_size           = chr_size,
  chr_name           = chr_name,
  normalization      = "NONE",
  norm_low           = norm_low,
  norm_high          = norm_high,
  downsampling_rates = 0.75,
  lambda             = 10,
  max_dist           = 0.01,
  nThread            = 4,
  alpha              = -0.25,
  max_iter           = 500
)

res <- res[order(res$frag_id), ]
res$start_bp <- (res$frag_id - 1) * frag_res
write.table(res[, c("start_bp","x","y","z")],
            paste0("coords_5kb_seed", SEED, ".tsv"),
            sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
cat("FLAMINGO_DONE wrote", nrow(res), "fragment coordinates\n")
