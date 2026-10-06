# code/ — training and evaluation

## Environment
Runs on Google Colab with a GPU. Dependencies are installed by the notebook: torch, torchvision, py7zr, torchmetrics[image], torch-fidelity, lpips, scikit-learn, noise, scipy, opencv-python, matplotlib. Locally there is no GPU and no data, so local runs are for syntax checks only.

## How to run
- Use the pipelines: `run_celeba.sh` (dataset 1) and `run_mnist.sh` (dataset 2). Each one `cd`s into this folder on the Drive and loops over `KAPPAS`.
- `run.py` reads six answers from stdin in this order: dataset (1 CelebA, 2 MNIST), model (1 CVAE, 2 Heavy-Tail VAE, 3 β-VAE, 4 Prior-VAE), κ (or β for models 3–4), latent dim, samples S, mode. The shell scripts pipe these in with `printf`. If you add or reorder prompts, update both shell scripts and `run_sampling_experiment.py`.
- Modes: 1 train, 2 eval set, 3 clean extracted data, 4 robustness reconstructions, 5 stochastic consistency, 6 t-SNE / 2D latent, 7 standard free energy (κ=0 metric).
- Pipeline order: train → eval set → `calculate_metrics.py <ds>` and `generate_corrupted_images.py <ds>` → mode 4 → `calculate_robustness_metrics.py <ds>` → zip eval set to Drive → modes 5 and 7 (MNIST also runs 2D models with modes 1 and 6, then mode 3 cleanup).
- Training resumes from `<model>_<param>_dim_<d>_latest.pth` if it exists. Delete or move the checkpoint to retrain from scratch. A NaN/Inf loss ends that κ run with exit 0, so the loop moves on. Look for `nan` rows in `epoch_log.txt`.

## Data provenance and paths
- CelebA raw: `celeba_dataset/celeba/img_align_celeba.zip` on Drive, extracted to `/content/celeba_dataset` (local Colab SSD). Random 70/15/15 split from `torch.randperm` with seed 42.
- MNIST raw: torchvision download (or `mnist_dataset/mnist_raw_data.zip` if present) to `/content/mnist_dataset`. The 60k train set is used for training. The 10k test set is split 5k validation / 5k test.
- Outputs: `<celeba_data|mnist_data>/<cvae|heavy_vae|beta_vae|prior_vae>/outputs_<kappa_X|beta_X>_dim_<d>_samples_<S>/` (checkpoint, `epoch_log.txt`, `batch_log.txt`, image grids, stochastic radii, standard free energy).
- Eval images are on the local SSD at `/content/evaluation_dataset/` (`originals/`, corruption folders, `reconstructions_<model>_<κ>_dim_<d>_samples_<S>` and `reconstructions_<corruption>_<model>_<κ>…`) and zipped to `<data>/evaluation_dataset.zip`. They get lost when the Colab session ends, so the zip is the only lasting copy.
- Metrics: `evaluation_results.txt` (FID, KID, LPIPS, MS-SSIM, PSNR, FRD, Prec, Rec) and `robustness_results_<corruption>.txt` per model folder. Both append and use JSON checkpoints, so rows can come out of κ order or duplicated.

## Results to paper figures
Metric tables in `tex files/main.tex` are copied from `evaluation_results.txt` and `robustness_results_*.txt`. Figures like `cvae_celeba_metrics_*.pdf`, `Fig_Energy.pdf` and `Beta_*.pdf` were made outside this repo, and their plotting scripts are not tracked. Save any new plotting script here and write the PDF into `tex files/`.

## Active models
- **CVAE (model 1)** is the current model: heavy-tail MCG latent + independent-equals scaling during training (σ/√(1+κ), κ̃ = κ/(1+κ)). Validation, eval and robustness sampling use the unscaled distribution (`scale_covariance=False`).
- **Heavy-Tail VAE (model 2)** is a benchmark: same loss, no IE scaling.
- **β-VAE (model 3)** is a benchmark that we will run soon on CelebA. It is also a baseline for the β-equivalence appendix, together with Prior-VAE (model 4, σ_p² = 1/√β).
- **κ convention:** `run.py` uses the paper's κ everywhere. The density is (1 + κQ/2)^−(1+κd/2)/κ, Z follows Eq. `partitionFunctionCoupledGaussian`, the sampler uses ν = 2/κ, and the domain is κ > −2/d. Don't bring back the 2κ form.
- Shared hyperparameters in `run.py`: Adam lr 5e-4, batch 64, 5 epochs, α=2, prior N(0,1), eval set 10,000 images, seed 42, no gradient clipping, no LR scheduler.
- CelebA: latent dim 100, S=5, κ ∈ {1e-6, 1e-4, 1e-2, 1, 10}. Only the cvae κ = 0 baseline is on Drive (its rows are kept in the results files). Every κ > 0 run must be redone with the fixed Z. `run_celeba.sh` currently has `MODEL_CHOICE=2` (heavy_vae).
- MNIST: latent dim 100 and 2, S=1, κ ∈ {0, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6}.

## Other files
- `run_sampling_experiment.py` imports from `run.py` with a fake-stdin hack and a hard-coded non-Colab data path. Its fake answers predate the dataset/model prompts, so it is stale.

## Project Map
- Root rules: `../CLAUDE.md`. Roadmap: `../TODO.md`.
- Past errors in this folder: `MISTAKES.md`. Keep this file and `MISTAKES.md` current and concise.
