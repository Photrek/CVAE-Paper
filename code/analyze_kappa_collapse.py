"""Term weights of the coupled loss per kappa, and batch logs put back on a common scale.

The coupled reconstruction loss is 0.5*M (+)_kappa A_x = A_x + 0.5*M*(1 + kappa*A_x), with M the squared error.
So the logged Recon and ELBO have a kappa-dependent offset (A_x) and weight (1 + kappa*A_x) and cannot be
compared across kappa. This script recovers M per pixel from the logged Recon.

Usage: python analyze_kappa_collapse.py [<data>/cvae/results_batch_log]
Standard library only, so it runs locally.
"""
import math
import os
import sys

IMAGE_DIM = 3 * 128 * 128  # CelebA
LATENT_DIM = 100


def log_partition(log_det, kappa, d):
    """Same as compute_log_partition_function in run.py (alpha = 2)."""
    if kappa < 1e-6:
        return 0.5 * log_det + 0.5 * d * math.log(2 * math.pi)
    return (0.5 * log_det + math.lgamma(1 / kappa) - math.lgamma(1 / kappa + d / 2)
            + d / 2 * math.log(2 * math.pi / kappa))


def normalization_term(log_det, kappa, d):
    """A = ln_kappa(Z^(1/(1 + kappa d/2))), as in run.py. At kappa = 0 it is ln Z."""
    log_term = log_partition(log_det, kappa, d) / (1 + kappa * d / 2)
    return log_term if kappa == 0 else math.expm1(kappa * log_term) / kappa


def weights(kappa):
    a_x = normalization_term(0.0, kappa, IMAGE_DIM)
    a_p = normalization_term(0.0, kappa, LATENT_DIM)
    w_recon = 1 + kappa * a_x  # weight on 0.5*M
    w_prior = 1 + kappa * a_p  # weight on 0.5*(|mu|^2 + trace)
    # The log-det force on each posterior variance v scales like v^p, against v for the trace term.
    # p near 1 leaves no restoring force, so the encoder can shrink to a point.
    p = kappa * LATENT_DIM / 2 / (1 + kappa * LATENT_DIM / 2)
    # Coupled KL of a point-mass posterior (mean and variance -> 0). Infinite at kappa = 0.
    kl_point = math.inf if kappa == 0 else 1 / kappa + a_p
    return a_x, w_recon, w_prior, p, kl_point


def print_weights():
    print(f"{'kappa':<8}{'A_x':>12}{'w_recon':>11}{'w_prior':>9}{'beta_eff':>10}{'p':>7}{'KL_point':>10}{'KL_point/w_recon':>18}")
    for kappa in [1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 10]:
        a_x, w_r, w_p, p, kl_pt = weights(kappa)
        print(f"{kappa:<8g}{a_x:12.4g}{w_r:11.4g}{w_p:9.4g}{w_p / w_r:10.4g}{p:7.3f}{kl_pt:10.4g}{kl_pt / w_r:18.4g}")


def print_batch_logs(folder):
    for name in sorted(os.listdir(folder)):
        if not name.startswith("batch_log_kappa_"):
            continue
        kappa = float(name.split("_")[3])
        a_x, w_r = weights(kappa)[:2]
        with open(os.path.join(folder, name)) as f:
            rows = [line.split("\t") for line in f.read().splitlines()[1:] if line.strip()]
        print(f"\n== {name}")
        print(f"{'batch':>6}{'ELBO':>12}{'KLD':>10}{'PostVar/dim':>13}{'MSE/pixel':>11}")
        for b in [0, 30, 60, 100, 1000, len(rows) - 1]:
            elbo, recon, kld, post_var = float(rows[b][3]), float(rows[b][4]), float(rows[b][5]), float(rows[b][7])
            mse = (recon - a_x) / (0.5 * w_r) / IMAGE_DIM
            print(f"{b:>6}{elbo:12.4g}{kld:10.4g}{post_var / LATENT_DIM:13.4g}{mse:11.4g}")


if __name__ == "__main__":
    print_weights()
    if len(sys.argv) > 1:
        print_batch_logs(sys.argv[1])
