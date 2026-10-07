# TODO

## Now
- Rerun every κ > 0 model after the partition-function fix (done 2026-10-06: Z in the paper's κ, mode 7 log1p(κQ/2), model name in reconstruction folders; see `code/MISTAKES.md`). All CelebA outputs were deleted, κ = 0 included, so all rows share the same loader and code.
  1. CelebA cvae: `run_celeba.sh` as is (`MODEL_CHOICE=1`, grid now includes κ = 0). Done 2026-10-06 on Colab L4. Resume restores the RNG state, and the corruption set is shared through `celeba_data/evaluation_dataset.zip`.
  2. CelebA heavy_vae: set `MODEL_CHOICE=2` and run again.
  3. MNIST cvae, d = 100 and d = 2: `run_mnist.sh` with S = 5 (`mnist_data/` is empty, so this is a fresh run).
  4. Check each `epoch_log_<run>.txt` (`results_epoch_log/` in the model folder) for `nan` rows, then copy the new numbers into the CelebA metric and robustness tables in `main.tex`, and redo `Fig_Energy.pdf` and `cvae_celeba_metrics_*.pdf` if they use κ > 0 runs.
- Done 2026-10-07: mode 7 now scores every model with one common Gaussian bound (lemma `lemma:common_gaussian_evaluation` in `main.tex`, Methods paragraph before "Architecture"). The old CelebA cvae `standard_free_energy.txt` was deleted (see `code/MISTAKES.md` for why the old metric was dropped). Next: rerun mode 7 on Colab for CelebA cvae (`run_celeba.sh` reruns only mode 7 because its rows are missing), then heavy_vae and MNIST. Check that `Decoder_Sigma` is close to the RMSE implied by PSNR (≈ 0.1) and that every row is finite. No mode 7 number is in the paper yet.
- Investigate CelebA cvae κ = 1e-2. It is worse than both neighbors: PSNR 15.85 dB, compared with 20.93 at κ = 1e-4 and 20.62 at κ = 1. FID is 202.78, compared with 116.39 at κ = 0 (`results_evaluation/evaluation_results.txt`). Mode 7 only scores the trained weights, so the cause is probably the training run. Check `results_epoch_log/epoch_log_kappa_0.01_dim_100_samples_5.txt` and the batch log for loss spikes, a stalled loss or a resume jump. Check its image grids too. If one run is unlucky, retrain it (same seed) and compare. κ = 10 (PSNR 9.32) may have the same problem or a real large-κ effect, so check it in the same way.
- Done 2026-10-06: resume/skip optimization (`code/pipeline_state.py`, `code/pipeline_lib.sh`). Tested on Colab (CelebA cvae): finished steps skip, and a fully done pipeline finishes in minutes.
- CelebA tables list κ up to 10⁶ plus a zoom grid (10^±0.5, 10^±1.5). `run_celeba.sh` only runs {0, 1e-6, 1e-4, 1e-2, 1, 10}. This will be updated later, after the reruns.
- Rewrite the stale "Architecture" subsection (it mentions TensorFlow, CIFAR and dense nets). The CNN description in Results is the correct one.

## Paper cleanup
- Done 2026-10-06: the four deferred math fixes (½ and loss sign in the standard-free-energy lemma, ι exponent in the simplified CFE, dx instead of dF/dq, divergence order p − q). `run.py` already matched; no numbers change. See `tex files/MISTAKES.md`.
- Done 2026-10-06: duplicate `figure.N`/`table.N` anchor warnings fixed (`float` now loads before `jmlr2e`/hyperref). Log is clean.
- Done 2026-10-06: text edits. Intro explains independent-equals as a fractal number of independent copies in the same state. Eq. `CE` is expanded to (1/κ)(1/∫f^ι − 1), with additivity (coupled sum) and extensivity. The closed-form subsection now ends by leading into the sampling step. Methods explains the S samples per input (Monte Carlo for the reconstruction term only, variance reduction). Appendix A ends with a summary of why the CVAE samples from q/^ι. Coauthors should review the wording.
- Done 2026-10-06: Methods frames S > 1 as non-ergodicity (training's per-step average vs the ensemble average over the posterior), kept with the variance argument. Appendix A renamed to second-order equivalence. The paper states S = 5 (Methods and CelebA setup), and `run_mnist.sh` now uses S = 5 too.
- Red notes still open: abstract, intro section outline, robustness Gaussian caption, crypto table and figure captions, sequential-means figure, latent-space section.
- Replace the "old results" MNIST section with current MNIST runs (t-SNE / 2D latent from modes 6 and 1 at d = 2). The old citations (`vanDerMaaten2008`, `1284395`, empty `\citep{}`) were removed from it; add the t-SNE and SSIM citations back when it is rewritten.
- Crypto section: the paragraph near line 1073 is unfinished ("nonzero, These results..."). The bullet list after the crypto figure is draft notes.

## Experiments
- Run the beta_vae benchmark on CelebA (model 3) and add it to the same tables.
- Latent-space analysis: MNIST neighbor graph and clustering scores. Decide whether to do the same for CelebA with its 40 attributes.
- Find or add the code for the crypto distribution fit (Hyvärinen score matching) and for any CVAE training on crypto returns. Neither is in this repo.
- Track the plotting scripts for `cvae_celeba_metrics_*.pdf`, `Fig_Energy.pdf` and `Beta_*.pdf` in `code/`.
- Fix or retire `code/run_sampling_experiment.py` (stale fake-stdin answers, non-Colab data path).

## Later
- Natural-gradient / information-geometric optimizer for the coupled exponential family (named as future work in the paper).
