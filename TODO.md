# TODO

## Now
- Rerun every κ > 0 model after the partition-function fix (done 2026-10-06: Z in the paper's κ, mode 7 log1p(κQ/2), model name in reconstruction folders; see `code/MISTAKES.md`). All CelebA outputs were deleted, κ = 0 included, so all rows share the same loader and code.
  1. CelebA cvae: `run_celeba.sh` as is (`MODEL_CHOICE=1`, grid now includes κ = 0). Started on Colab L4, 2026-10-06. Resume restores the RNG state, and the corruption set is shared through `celeba_data/evaluation_dataset.zip`.
  2. CelebA heavy_vae: set `MODEL_CHOICE=2` and run again.
  3. MNIST cvae, d = 100 and d = 2: `run_mnist.sh` (`mnist_data/` is empty, so this is a fresh run).
  4. Check each `epoch_log.txt` for `nan` rows, then copy the new numbers into the CelebA metric and robustness tables in `main.tex`, and redo `Fig_Energy.pdf` and `cvae_celeba_metrics_*.pdf` if they use κ > 0 runs.
- CelebA tables list κ up to 10⁶ plus a zoom grid (10^±0.5, 10^±1.5). `run_celeba.sh` only runs {0, 1e-6, 1e-4, 1e-2, 1, 10}. This will be updated later, after the reruns.
- Rewrite the stale "Architecture" subsection (it mentions TensorFlow, CIFAR and dense nets). The CNN description in Results is the correct one.

## Paper cleanup
- Undefined citations: `vanDerMaaten2008`, `1284395` (old MNIST section).
- Red notes still open: abstract, intro section outline, robustness Gaussian caption, crypto table and figure captions, sequential-means figure, latent-space section.
- Replace the "old results" MNIST section with current MNIST runs (t-SNE / 2D latent from modes 6 and 1 at d = 2).
- Crypto section: the paragraph near line 1073 is unfinished ("nonzero, These results..."). The bullet list after the crypto figure is draft notes.
- Author heading: "Igor Oliveria" typo in `\jmlrheading`. Thistleton is in the heading but not in the author blocks. Addresses are missing.

## Experiments
- Run the beta_vae benchmark on CelebA (model 3) and add it to the same tables.
- Latent-space analysis: MNIST neighbor graph and clustering scores. Decide whether to do the same for CelebA with its 40 attributes.
- Find or add the code for the crypto distribution fit (Hyvärinen score matching) and for any CVAE training on crypto returns. Neither is in this repo.
- Track the plotting scripts for `cvae_celeba_metrics_*.pdf`, `Fig_Energy.pdf` and `Beta_*.pdf` in `code/`.
- Fix or retire `code/run_sampling_experiment.py` (stale fake-stdin answers, non-Colab data path).

## Later
- Natural-gradient / information-geometric optimizer for the coupled exponential family (named as future work in the paper).
