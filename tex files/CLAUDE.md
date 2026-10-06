# tex files/ — the paper

## Build
- `latexmk -pdf -interaction=nonstopmode main.tex` from this folder (pdflatex + bibtex). Send build output to the scratchpad with `-outdir=<scratch>` to keep aux files out of the Drive. They are gitignored anyway.
- After a build, check the log for `undefined` citations/references and multiply-defined labels.
- Log cleanup done 2026-10-06. Fixed: empty underfull line after each proof (`proof` redefined in `main.tex`, since `jmlr2e.sty` stays untouched), caption-package notice (`subfig` with `caption=false`), long URL overfull. Still open and cosmetic: two `Overfull \vbox` on the training algorithm page and the crypto figure page (harmless with `\raggedbottom`), and a few underfull lines in the reference list.

## Template
- `jmlr2e.sty` with options `[abbrvbib, preprint]`. Bibliography style is abbrvnat via natbib. Use `\citep` for parenthetical and `\citet` for textual citations.
- Heading: `\jmlrheading`, `\ShortHeadings`, author blocks with `\name/\email/\addr`. Don't edit `jmlr2e.sty`.
- The preamble loads several packages twice (graphicx, tikz, xcolor). Don't add more duplicates.

## Citations
- One bib file: `MICS_BibTex.bib` (exported from a reference manager, keys like `nelsonUniqueUniversalEntropy2026`). Add new entries there and keep the existing key style. Never invent a reference. If a source is unknown, leave a red note and ask.

## Notation and math style
- κ is the nonlinear statistical coupling, the "source of nonlinearity" (shape divided by stretching α). For the coupled Gaussian, α = 2. d is the latent dimension, S the number of samples per input.
- Coupled log `\ln_\kappa`, coupled exponential `\exp_\kappa`, coupled sum `\oplus_\kappa`, independent-equals distribution `f/^{\iota(\kappa,d_2m)}`, coupled entropy `H_\kappa`, Coupled Free Energy `\mathcal{F}_{\theta,\phi,\kappa}`.
- Independent-equals of a coupled Gaussian: κ̃ = κ/(1+κ) and Σ̃ = Σ/(1+κ), so σ̃ = σ/√(1+κ). The code uses the same.
- Coupled Gaussian exponent is −(1+κd/2). The coupled log of the density uses the power −1/(1+κd/2).
- Don't put `\sfrac` in subscripts (it asks for a 3.8 pt font that OT1 lacks). Use `\kappa/a`. Wrap math in section titles with `\texorpdfstring`. Use `[ht!]`, not `[h!]`.
- Vectors in bold (`\mathbf{x}`, `\boldsymbol{\mu}`). Use `\top` or `\intercal` for transpose, one per equation.
- Proofs go in the appendices as lemmas/theorems with labels. The main text cites them by `\ref`.

## Figures and tables
- Figure files are in this folder (PDF preferred, PNG for raster). `\graphicspath{{./images/}}` covers older MNIST images in `images/`.
- Label style: `fig:...`, `tab:...`, `alg:...`, `eq:...`. Older labels end in `.png`. Keep them, but every label must be unique.
- Metric tables use `booktabs`, a `gray!20` header row, `\textcolor{blue}` for better-than-baseline and `\best{}` (bold green) for the best per column. The κ = 0 row is the baseline.
- Table numbers come from `../celeba_data/<model>/evaluation_results.txt` and `robustness_results_*.txt`. Uncertainty is written as a single digit in parentheses.
- `\textcolor{red}{...}` marks open author notes. Remove one only after the note has been addressed.

## Project Map
- Root rules: `../CLAUDE.md`. Roadmap: `../TODO.md`. Code that produces the numbers: `../code/CLAUDE.md`.
- Past errors in this folder: `MISTAKES.md`. Keep this file and `MISTAKES.md` current and concise.
