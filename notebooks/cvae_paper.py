# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # CVAE Paper: Colab launcher
# Source of truth for `CVAE_PAPER.ipynb`. Edit this file, then regenerate the notebook with
# `jupytext --to ipynb --update notebooks/cvae_paper.py -o notebooks/CVAE_PAPER.ipynb`.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Install all required external libraries
# !pip install py7zr "torchmetrics[image]" torch-fidelity lpips scikit-learn noise scipy opencv-python

# %%
# %cd "/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/code"

# !chmod +x *.sh
# !sed -i 's/\r$//' *.sh

# %% [markdown]
# ## CelebA pipelines
# `MODEL_CHOICE` picks the model: 1 CVAE, 2 Heavy-Tail VAE, 3 β-VAE, 4 Prior-VAE. `run_celeba.sh` uses the κ grid
# for models 1-2 and the β grid for models 3-4. Finished steps are skipped, so a done model costs only a few seconds.
# A failure in one cell does not stop the next one, since `!` commands do not raise errors in Colab.

# %%
# !MODEL_CHOICE=2 ./run_celeba.sh

# %%
# !MODEL_CHOICE=3 ./run_celeba.sh


# %% [markdown]
# ## Free the GPU
# The next cell disconnects and deletes the runtime, so it stops using GPU credits once the pipeline ends.
# Run all cells with it at the end. To get a pop-up when the run finishes, turn on Colab's
# Tools > Settings > Notifications ("Notify me when long-running executions complete").

# %%
from google.colab import runtime
runtime.unassign()
