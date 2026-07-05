#!/usr/bin/env Rscript
# Real FLAMINGO (Wang et al. 2022) reconstruction of GM12878 chr22 (hg19) at 50 kb,
# from the SAME KR-normalized contacts the GenoFold benchmark uses.
# Run from this directory (FLAMINGO writes Domain_data/ Genomic_loc/ into CWD).

suppressPackageStartupMessages({
  library(FLAMINGOr); library(Matrix); library(data.table); library(mgcv)
  library(progress); library(parallel)
})

chr_name   <- "chr22"
chr_size   <- 51304566        # chr22 hg19, bp
domain_res <- 1e6
frag_res   <- 5e4             # 50 kb

# 3-col whitespace file: bp_start1  bp_start2  KR_value  (symmetric, both triangles).
contact_file <- "chr22_50kb.contacts.txt"

# FLAMINGO re-normalizes as value/(norm[i]*norm[j]); our values are ALREADY KR-normalized,
# so we pass all-ones norm vectors (built alongside the contacts) => values pass through.
norm_high <- "ones_high.txt"  # length ceil(chr_size/5e4) = 1027
norm_low  <- "ones_low.txt"   # length ceil(chr_size/1e6) = 52

SEED <- as.integer(Sys.getenv("FSEED","0")); set.seed(SEED)
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
res$start_bp <- (res$frag_id - 1) * frag_res   # 0-based 50 kb bin start -> maps to GenoFold bin
# write (start_bp, x, y, z) to match experiments/flamingo_real.py
write.table(res[, c("start_bp","x","y","z")],
            paste0("coords_seed",SEED,".tsv"),
            sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
cat("FLAMINGO_DONE wrote", nrow(res), "fragment coordinates\n")
