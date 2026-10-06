# CVAE Paper

## Goal
Write and back with experiments the JMLR-style paper "Stable training of extreme models using the Coupled Variational Autoencoder". The CVAE replaces the VAE free energy with the Coupled Free Energy and samples the latent space from the independent-equals distribution, so training stays finite even for very large coupling κ.

## Shared stack
- Paper: LaTeX in `tex files/` (jmlr2e template, natbib).
- Experiments: PyTorch scripts in `code/`, run on Google Colab (GPU) through the launcher in `notebooks/`.
- Datasets: CelebA (128×128 RGB) and MNIST (32×32 grayscale). The paper also cites cryptocurrency log-returns, but that code is not in this repo.

## Rules that apply everywhere
- Colab mounts the Shared Drive at `/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper`. The scripts use absolute paths built from that root, so a renamed or moved folder breaks Colab runs.
- `celeba_data/`, `celeba_dataset/` and `mnist_data/` stay at the project root. They are gitignored, large, and hold raw data plus training outputs. Never commit them, and never delete or overwrite them without asking first.
- Reproducibility: the seed is fixed at 42 in `code/run.py`. Don't change the seed, the data split or the evaluation-set size silently, because every reported number depends on them.
- Numbers in the paper must trace back to a results file under `celeba_data/` or `mnist_data/`. Don't type in or "fix" a reported number by hand.
- Commit only when I ask.

## Project-wide lessons
- The paper and code have drifted apart before (latent dimension, gradient clipping, κ grid). When you change one side, check the other.

## Writing style
When I ask you to write text, use simple words and first-principles logic. Avoid AI buzzwords and absolutist words (perfectly, always, exactly). Don't overuse bullet points or dashes. Be concise and clear. This applies to code comments, TeX writing, notebook markdown cells, and so on.

## Memory-cycle protocol
Keep the root `CLAUDE.md`, the folder `CLAUDE.md` files, the folder `MISTAKES.md` files and the root `TODO.md` up to date as part of finishing a task, not as a separate follow-up, and tell me which files you updated. Sessions get cleared and picked back up cold, so these files are the only handoff.
- After completing a stage of a staged plan, record the result and remove or mark it done in `TODO.md` before ending the turn.
- When we finish a subtask, give me a prompt I can paste to continue in a new session.
- After empirically rejecting an approach, add it to the `MISTAKES.md` of the folder it belongs to.
- After a config or hyperparameter decision that future work must not silently override, record it in the "Active models" section of `code/CLAUDE.md`.
- Make these updates in the same turn as the work that produced the finding, never as a deferred TODO. Keep every file concise and objective. Remove instructions that no longer hold.

## Project Map
- `tex files/CLAUDE.md`: paper build, notation, figures, citations. Past errors in `tex files/MISTAKES.md`.
- `code/CLAUDE.md`: training/evaluation pipeline, data flow, Active models. Past errors in `code/MISTAKES.md`.
- `notebooks/CLAUDE.md`: Colab launcher, `.py` → `.ipynb` rule. Past errors in `notebooks/MISTAKES.md`.
- `TODO.md`: the single roadmap and cold-start resume point.
- `ExperimentSamples/`: scratch scripts. Ignore unless I ask.
