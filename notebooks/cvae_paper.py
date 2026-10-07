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

# %%
# !./run_celeba.sh


# %% [markdown]
# ## Free the GPU
# The next cell disconnects and deletes the runtime, so it stops using GPU credits once the pipeline ends.
# Run all cells with it at the end. To get a pop-up when the run finishes, turn on Colab's
# Tools > Settings > Notifications ("Notify me when long-running executions complete").

# %%
from google.colab import runtime
runtime.unassign()
