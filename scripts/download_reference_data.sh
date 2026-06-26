#!/bin/bash
# download_reference_data.sh
# ----------------------------
# Downloads:
#   1. hg19 chr22 reference genome FASTA (for real GC content + sequences)
#   2. GM12878 DNase-seq bigWig signal (for real NGS coverage)
#
# Both are small, fast downloads (~15-50 MB total) compared to the
# multi-GB Hi-C file, and both are freely available with no login.
#
# Sources:
#   - Reference genome: UCSC Genome Browser (hg19, Feb 2009 GRCh37)
#   - DNase-seq signal:  ENCODE / UCSC (wgEncodeUwDnase, GM12878)
#
# Usage:
#   bash scripts/download_reference_data.sh

set -e

REF_DIR="data/reference"
mkdir -p "$REF_DIR"

echo "==========================================================="
echo "  Downloading reference genome (hg19 chr22)"
echo "==========================================================="
FASTA_GZ="$REF_DIR/chr22.fa.gz"
FASTA="$REF_DIR/chr22.fa"

if [ -f "$FASTA" ]; then
    echo "Reference FASTA already exists at $FASTA -- skipping."
else
    curl -L -o "$FASTA_GZ" \
        "https://hgdownload.cse.ucsc.edu/goldenpath/hg19/chromosomes/chr22.fa.gz" \
        --progress-bar
    echo "Decompressing..."
    gunzip "$FASTA_GZ"
    echo "Reference genome saved: $FASTA"
fi

echo ""
echo "==========================================================="
echo "  Downloading GM12878 DNase-seq coverage (bigWig)"
echo "==========================================================="
BIGWIG="$REF_DIR/GM12878_DNase.bigWig"

if [ -f "$BIGWIG" ]; then
    echo "DNase-seq bigWig already exists at $BIGWIG -- skipping."
else
    curl -L -o "$BIGWIG" \
        "https://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeUwDnase/wgEncodeUwDnaseGm12878RawRep1.bigWig" \
        --progress-bar
    echo "DNase-seq coverage saved: $BIGWIG"
fi

echo ""
echo "==========================================================="
echo "  Download complete!"
echo "==========================================================="
echo "  Reference FASTA: $FASTA"
echo "  DNase-seq bigWig: $BIGWIG"
echo ""
echo "Next step: run the pipeline with both real inputs:"
echo "    pip install pyfaidx pyBigWig"
echo "    python3 scripts/run_real_data.py --chrom chr22 --bins 200 \\"
echo "        --fasta data/reference/chr22.fa \\"
echo "        --bigwig data/reference/GM12878_DNase.bigWig"
