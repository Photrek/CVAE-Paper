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

# !chmod +x run_celeba.sh
# !sed -i 's/\r$//' run_celeba.sh

# %%
# !./run_celeba.sh
