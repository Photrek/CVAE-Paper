# notebooks/ — Colab launchers

## Source of truth
- `cvae_paper.py` (jupytext percent format) is the file to read and edit. Don't read or edit `CVAE_PAPER.ipynb` directly. It costs far more tokens and is a generated artifact.
- After every edit to a `.py` here, regenerate its notebook in the same turn so I can run it in Colab:
  `jupytext --to ipynb --update notebooks/cvae_paper.py -o notebooks/CVAE_PAPER.ipynb`
  `--update` keeps the Colab metadata (GPU accelerator) and existing outputs. jupytext is installed in `~/.virtualenvs/base` (`pip install jupytext` if it is missing).
- Cells start with `# %%`, and markdown cells with `# %% [markdown]`. Shell and magic lines are written commented (`# !pip ...`, `# %cd ...`), and jupytext uncomments them in the `.ipynb`.

## What the notebook does
Mounts Drive, installs dependencies, `cd`s into `CVAE Paper/code`, fixes CRLF line endings on all `.sh` files (`pipeline_lib.sh` is sourced by both pipelines), runs `MODEL_CHOICE=2 ./run_celeba.sh` and then `MODEL_CHOICE=3 ./run_celeba.sh` (heavy_vae, then beta_vae; finished steps skip, and a failing `!` cell does not stop the next one), then a final cell calls `runtime.unassign()` to free the GPU (it ends the session, so run it last; Colab's Tools > Settings > Notifications gives the finish pop-up). The model is set per cell. The dataset and the κ/β grids live in the shell scripts in `../code/`, not here.

## Project Map
- Root rules: `../CLAUDE.md`. Roadmap: `../TODO.md`. Pipeline details: `../code/CLAUDE.md`.
- Past errors in this folder: `MISTAKES.md`. Keep this file and `MISTAKES.md` current and concise.
