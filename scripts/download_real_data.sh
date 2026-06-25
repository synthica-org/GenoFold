#!/bin/bash
# download_real_data.sh
# ----------------------
# Downloads a real, publicly available Hi-C dataset from GEO
# (Gene Expression Omnibus) — no account or login required.
#
# Dataset: GM12878 in-situ Hi-C (MboI digestion)
# Source:  GEO Series GSE63525 (Rao et al. 2014, Cell)
# This is one of the most cited Hi-C datasets in the field --
# the original "in-situ Hi-C" method paper.
#
# Usage:
#   bash scripts/download_real_data.sh

set -e

OUTPUT_DIR="data/real"
mkdir -p "$OUTPUT_DIR"

URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary.hic"
OUTPUT_FILE="$OUTPUT_DIR/GM12878_insitu_hic.hic"

echo "==========================================================="
echo "  Downloading GM12878 Hi-C data from GEO (GSE63525)"
echo "==========================================================="
echo "  Source:      NCBI GEO (Rao et al. 2014, Cell)"
echo "  Accession:   GSE63525"
echo "  Destination: $OUTPUT_FILE"
echo "==========================================================="
echo ""
echo "NOTE: This file is large (multi-GB). Download time depends"
echo "      on your internet connection."
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo "File already exists at $OUTPUT_FILE -- skipping download."
else
    curl -L -o "$OUTPUT_FILE" "$URL" --progress-bar
fi

echo ""
echo "Download complete: $OUTPUT_FILE"
echo ""
echo "Next step: install hic-straw, then run the pipeline:"
echo "    pip install hic-straw"
echo "    python3 scripts/run_real_data.py --hic data/real/GM12878_insitu_hic.hic --format hic"
