# GenoFold — IEEE BIBM 2026 UGHS submission package

LaTeX source for *"GenoFold: A Confidence-Aware, Explainable, Multi-Modal
Framework for 3D Genome Architecture Reconstruction"*, formatted with
`IEEEtran` in `conference` mode (the IEEE Computer Society proceedings format
required by the symposium).

## Files

| File | Purpose |
|---|---|
| `main.tex` | The paper. This is the only file you normally edit. |
| `references.bib` | Bibliography database. |
| `main.bbl` | Pre-built bibliography — lets the paper compile without running BibTeX. |
| `IEEEtran.cls`, `IEEEtran.bst` | Bundled so the project builds on any TeX install. |
| `figures/` | Drop figure PDFs/PNGs here. |
| `main.pdf` | Current compiled output (4 pages). |
| `Makefile` | `make` to build, `make clean` to remove aux files. |

## Folder layout

| Path | Contents |
|---|---|
| `main.tex` / `main.pdf` | Submitted version (from the uploaded package). |
| `ughs.tex` / `ughs.pdf` | Camera-ready revision: fills the 5-page limit, responds to both reviewer reports. |
| `figures/` | All figure PNGs (`fig_*.png` plots, `pipeline_image*.png` diagrams). |
| `archive/` | Earlier long draft `GenoFold_BIBM26.tex` and its `tables/`. |

### How `ughs.tex` responds to the reviews

- Sparse result is now credited to the learned prior, not to multi-modality, using the downsampling ablation (+0.005 dSCC). The missing Hi-C-only GenoFold and HiC-GNN sparse runs are named as open experiments.
- "Confidence"/"calibrated" removed from the title and text. The decile reliability curve is reported (flat for deciles 1–9, rising only in the top decile). The missing coverage and contact-count-proxy tests are stated.
- Attribution results added (IG: GC/mappability 0.43, coverage 0.30, position 0.18, 4-mers 0.10), framed as sensitivity rather than necessity.
- FLAMINGO taken out of the main table; 5 kb numbers given in the text.
- GenoFold per-chromosome dSCC (chr22/20/16) reported. The baseline-per-chromosome table is listed as not yet run.
- Species embedding declared inactive and not counted as part of the method.

## Building

```bash
latexmk -pdf main.tex     # easiest
make                      # same thing
```

Or manually:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Overleaf: upload the whole folder and set `main.tex` as the main document. It
works as-is; no package installation needed.

## Before you submit — checklist

1. **Author block** (top of `main.tex`). Replace the placeholder names, emails,
   and institutions. The symposium requires the affiliation to state whether the
   first author is a **high school** or **undergraduate** student — the line
   currently reads `High School Student`; change it if that is wrong.
2. **Page limit.** High school first author: **5 pages**. Undergraduate first
   author: **6 pages**. Everything counts — figures, tables, references. The
   paper currently compiles to **4 pages**, leaving room for roughly two
   single-column figures.
3. **Figures.** Two figure blocks are commented out near the end of the Results
   section (resolution comparison, downsampling curve). Add the image files to
   `figures/` and uncomment them.
4. **Review is single-blind**, so author names stay visible. Do not anonymize.
5. Submit through the portal linked from
   <https://bibm2026-hs.github.io/> (CyberChair).

## Notes on changes made to the source draft

- **Reference numbering was misaligned in the draft and has been corrected.**
  The draft cited `[7]` for ShRec3D but listed Imakaev et al. (the ICE
  normalization paper) there, and cited `[8]` for ICE but listed ChromHMM.
  Lesne et al. (2014), the actual ShRec3D paper, has been added; ICE now points
  to Imakaev; the unused ChromHMM entry was dropped.
- The draft cited `[18]` both for the chromatin-tracing distance matrices and
  for the GenoFold GitHub repository. Bintu et al. (2018), *Science* — the
  chromatin-tracing study that measured IMR90 and K562 chr21 — has been added
  as a separate entry. **Please confirm this is the dataset you used.**
- Gal & Ghahramani was listed but never cited; it is now cited where Monte Carlo
  dropout is introduced.
- The draft referenced a "Table VI" and "Table III" that were not included, and
  Figures 1 and 2 that were not supplied. Those dangling pointers were rewritten
  as prose so nothing compiles to a broken reference. Add the figures back if
  you have them.
- The loss equation was floating between subsections II-G and II-H; it now sits
  inside the Training Objective subsection where it belongs.
- Light grammar fixes only, no changes to any number or claim: "exceed" →
  "exceeds", "biology expected" → "biologically expected", "the contact graph
  thinks" → "thins", and the garbled Discussion clause "its node features are
  learned prior to not depend on a dense contact graph" → "its node features and
  learned prior do not depend on a dense contact graph".
- Two draft sentences referenced content that was cut ("see disclaimer",
  "omitted for space, available in supplementary"). Reworded, since there is no
  supplementary channel described for this venue.
