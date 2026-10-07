# tex files/ — the paper

## Build
- `latexmk -pdf -interaction=nonstopmode main.tex` from this folder (pdflatex + bibtex). Send build output to the scratchpad with `-outdir=<scratch>` to keep aux files out of the Drive. They are gitignored anyway.
- After a build, check the log for `undefined` citations/references and multiply-defined labels.
- With `-outdir`, latexmk can silently reuse the stale `main.bbl` in this folder and skip bibtex. For a clean check, run pdflatex, then `bibtex main` inside the outdir with `BIBINPUTS`/`BSTINPUTS` pointing here, then pdflatex twice.
- Log is clean as of 2026-10-06 (no warnings, no Overfull/Underfull). Fixes: empty underfull line after each proof (`proof` redefined in `main.tex`, since `jmlr2e.sty` stays untouched), caption-package notice (`subfig` with `caption=false`), long URL, training algorithm in `\small`, `Fig_Sequential_means.png` capped at 0.45 of the text height, `\raggedright` around `\bibliography`. Rebuild and recheck the log after edits.

## Template
- `jmlr2e.sty` with options `[abbrvbib, preprint]`. Bibliography style is abbrvnat via natbib. Use `\citep` for parenthetical and `\citet` for textual citations.
- Heading: `\jmlrheading`, `\ShortHeadings`, author blocks with `\name/\email/\addr`. Don't edit `jmlr2e.sty`.
- The preamble loads several packages twice (graphicx, tikz, xcolor). Don't add more duplicates.
- `float` is loaded before `jmlr2e`, because `jmlr2e` loads hyperref and hyperref patches floats only if `float` is already there. Loaded after it (directly or through `algorithm`), every `figure.N`/`table.N` anchor is written twice and pdfTeX warns "destination with the same identifier". Any package that defines float types goes before `jmlr2e` too.

## Citations
- One bib file: `MICS_BibTex.bib` (exported from a reference manager, keys like `nelsonUniqueUniversalEntropy2026`). Add new entries there and keep the existing key style. Never invent a reference. If a source is unknown, leave a red note and ask.

## Notation and math style
- The basis paper is `entropy.tex` (Nelson, unique universal entropy). It is nearly final. `main.tex` follows its notation. Don't edit `entropy.tex`.
- κ is the nonlinear statistical coupling, the "source of nonlinearity". The asymptotic tail shape is κ/α. For the coupled Gaussian, α = 2. d is the latent dimension, d_d the data dimension (3×128×128 for CelebA), S the number of samples per input. Reconstruction terms use d_d, latent terms use d (the code does the same with `data_space_dim`).
- Information dimension ι(κ,α,d) = 1 + ακ/(α+dκ), written `\iota(\kappa,2,d)` (Eq. `equ_infoDim`). Independent-equals distribution `f/^{\iota(\kappa,2,d)}`. Always write "independent-equals" with the hyphen. Relative risk aversion is ι(κ,2,d) − 1 = κ/(1+κd/2). Don't bring back R or d_m.
- Coupled entropy as in the basis paper: `\frac{1}{2}H(\mathbf{X};\kappa,2,d)^2` equals the independent-equals integral (Eq. `CE`), which reduces to (1/κ)(1/∫f^ι dx − 1) (checked numerically). Composition: ½H² of independent systems combines by the coupled sum (Eq. `coupledSum`, u⊕v = u+v+κuv). Extensivity: H/N → 1 for W(N) ~ exp_κ(N²/2)^{1+κd/2}. Section labels: `sec:cvae`, `sec:methods`. S = 5 for both datasets, stated in Methods; keep it in sync with `../code/run_*.sh`. Methods motivates S > 1 by non-ergodicity of heavy-tailed systems plus variance reduction. Appendix A is a second-order equivalence (the first-order terms cancel). The CFE terms are the coupled reconstruction loss `\mathcal{L}_\kappa(\mathbf{x})` and coupled divergence `D_\kappa(q\|p)`, not H.
- Partition function `Z(\boldsymbol{\Sigma},\kappa)` everywhere (Eq. `partitionFunctionCoupledGaussian`). The independent-equals one is the same function, Z(Σ̃,κ̃). Shorthands Z_q, Z_p, Z_{x|z} are fine inside proofs.
- Σ is the scale matrix, always `\boldsymbol{\Sigma}` (plain only in the 1D figure caption). Never call it covariance for κ > 0. The second independent-equals moment returns Σ.
- Coupled log `\ln_\kappa`, coupled exponential `\exp_\kappa`, coupled sum `\oplus_\kappa`, Coupled Free Energy `\mathcal{F}_{\theta,\phi,\kappa}`.
- Independent-equals of a coupled Gaussian: κ̃ = κ/(1+κ) and Σ̃ = Σ/(1+κ), so σ̃ = σ/√(1+κ). The code uses the same.
- Independent-equals averages integrate against dx (dz for latents), never dF or dq/^ι, since f/^ι already holds the density. Coupled divergence is E_{q/^ι}[ln_κ p^{…} − ln_κ q^{…}] (→ KL at κ = 0). Reconstruction loss is ln_κ p^{−1/(1+κd_d/2)} (→ −log p).
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
