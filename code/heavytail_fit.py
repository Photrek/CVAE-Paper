"""Coupled Gaussian fit to the chosen heavy-tail datasets (the analogue of cryptosData_fit.py).

Run with  python3 -I code/heavytail_fit.py  (the folders in datasets/ are untrusted downloads, made by heavytail_download.py).

Input form of each dataset (decided 2026-10-09):
  event data (storm, EAGLE-I, GCMT seismic moment, Norwegian fire, GOES, data-breach sizes): the positive value divided by its
      median, times a random sign (seed 42). The tail index is the same as that of the positive value.
  time series (Wikipedia Trump, Guadalupe river at Spring Branch TX): the signed difference of consecutive records, centered at the median
      and divided by the MAD.
Fit (decided 2026-10-13): mu = median, and sigma, kappa minimize the squared error of log P(|X - mu| > t) on 300 points
spaced evenly in log t, so the body and the tail weigh the same. The table also gives the Hill index (k = HILL_K), the
tail-anchored fit (kappa = 2/alpha_Hill, sigma matched at the k-th largest value) and the ML kappa for comparison.
The infinite-variance claim rests on alpha_Hill < 2.

Writes heavytail_results/ at the project root (gitignored): fit_table.tsv, fit_table.md and Fig_HeavyTail_Distro.png
(density of |x - median| from log-spaced bins, on log-log axes, one panel per dataset labeled (a), (b), ...).
No model is trained.
"""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq, minimize
from scipy.special import betainc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "heavytail_results")
os.makedirs(OUT, exist_ok=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cf = _load("cryptosData_fit")        # log_pdf, fit_ml, hill_alpha
hp = _load("heavytail_pilot")        # dataset loaders; `-I` removes the script folder from sys.path, so load by path

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "axes.labelsize": 22, "axes.titlesize": 20,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 18, "savefig.dpi": 200,
})

SEED = 42
MAX_FIT = 200_000
HILL_K = 200
N_GRID = 300
# label -> (loader name in heavytail_pilot.CANDIDATES, "event" or "series")
DATASETS = {
    "Storm property damage": ("storm_property_damage", "event"),
    "EAGLE-I customer-hours": ("eaglei_customer_hours", "event"),
    "Earthquake moment": ("gcmt_moment", "event"),
    "Norwegian fire": ("norwegian_fire", "event"),
    "Wikipedia (Trump) views": ("wiki_Donald_Trump", "series"),
    "GOES flare peak": ("goes_flare_peak", "event"),
    "Guadalupe river flow": ("nwis_08167500_GuadalupeSpringBranch", "series"),
    "Health-data breach size": ("hhs_breach_individuals", "event"),
}


def make_input(name, kind):
    """Standardized signed 1D array for one dataset."""
    parts = hp.build(name)
    if kind == "event":
        x = parts["raw"]
        x = x[np.isfinite(x) & (x > 0)]
        sign = np.random.default_rng(SEED).choice([-1.0, 1.0], size=len(x))
        return sign * x / np.median(x)
    x = np.round(parts["diff"], 6)     # records have at most 2 decimals; rounding removes float residues like 1e-16
    x = x[np.isfinite(x)]
    mad = float(np.median(np.abs(x - np.median(x))))
    return (x - np.median(x)) / mad


def survival(t, fit):
    """P(X - mu > t) of the coupled Gaussian: a Student-t with nu = 2/kappa, so the two-sided tail is an incomplete beta."""
    k = fit["kappa"]
    s = np.asarray(t) / fit["sigma"]
    return 0.5 * betainc(1 / k, 0.5, 1 / (1 + k * s * s / 2))


def _finish(z, sigma, kappa):
    mu = float(np.median(z))
    return {"mu": mu, "sigma": float(sigma), "kappa": float(kappa), "ll": float(np.mean(cf.log_pdf(z, mu, sigma, kappa)))}


def fit_tail_anchored(z, alpha, k=HILL_K):
    """kappa = 2/alpha (Hill) and sigma chosen so the model matches the empirical P(|X| > t0) = k/n at the k-th largest |z|."""
    a = np.sort(np.abs(z - np.median(z)))[::-1]
    t0, kappa, target = a[k], 2 / alpha, k / len(z)
    f = lambda ls: 2 * survival(t0, {"sigma": math.exp(ls), "kappa": kappa}) - target
    return _finish(z, math.exp(brentq(f, -20, 20)), kappa)


def survival_grid(z):
    """Empirical log P(|X - med| > t) at N_GRID values of t spaced evenly in log t (t > 0)."""
    a = np.sort(np.abs(z - np.median(z)))
    n = len(a)
    logsf = np.log(1 - np.arange(1, n + 1) / (n + 1))
    idx = np.clip(np.searchsorted(a, np.geomspace(a[a > 0].min(), a[-1], N_GRID)), 0, n - 1)
    return a[idx], logsf[idx]


def log_model_sf(t, sigma, kappa):
    return np.log(np.maximum(2 * survival(t, {"sigma": sigma, "kappa": kappa}), 1e-300))


def rmse_sf(z, fit):
    tx, ty = survival_grid(z)
    return float(np.sqrt(np.mean((log_model_sf(tx, fit["sigma"], fit["kappa"]) - ty) ** 2)))


def fit_whole_curve(z, start):
    """sigma and kappa that minimize the squared error of log P(|X| > t) on the log-spaced grid, from `start`."""
    tx, ty = survival_grid(z)
    loss = lambda p: np.mean((log_model_sf(tx, math.exp(p[0]), math.exp(p[1])) - ty) ** 2)
    r = minimize(loss, [math.log(start["sigma"]), math.log(start["kappa"])], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
    return _finish(z, math.exp(r.x[0]), math.exp(r.x[1]))


def main():
    rows, data = [], {}
    for label, (name, kind) in DATASETS.items():
        z = make_input(name, kind)
        n = len(z)
        zf = z if n <= MAX_FIT else z[np.random.default_rng(SEED).choice(n, MAX_FIT, replace=False)]
        ml = cf.fit_ml(zf)
        alpha = cf.hill_alpha(z, HILL_K)
        hl = fit_tail_anchored(z, alpha)
        wc = fit_whole_curve(z, hl)
        data[label] = (z, wc, hl, alpha)
        rows.append({"dataset": label, "loader": name, "input": kind, "n": n,
                     "alpha_hill": alpha, "alpha_hill_k50": cf.hill_alpha(z, 50), "alpha_hill_k1000": cf.hill_alpha(z, 1000),
                     "kappa_fit": wc["kappa"], "sigma_fit": wc["sigma"], "rmse_fit": rmse_sf(z, wc),
                     "kappa_hill": hl["kappa"], "sigma_hill": hl["sigma"], "rmse_hill": rmse_sf(z, hl),
                     "kappa_ml": ml["kappa"], "frac_zero": float(np.mean(z == np.median(z))),
                     "mean": float(np.mean(z)), "var": float(np.var(z)),
                     "var_infinite_hill": "yes" if alpha < 2 else "no", "mean_infinite_hill": "yes" if alpha < 1 else "no"})
        print(f"{label:26s} n={n:8d} alpha={alpha:.2f} kappa_fit={wc['kappa']:.2f} rmse_fit={rows[-1]['rmse_fit']:.2f} "
              f"kappa_hill={hl['kappa']:.2f} rmse_hill={rows[-1]['rmse_hill']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "fit_table.tsv"), sep="\t", index=False, float_format="%.4g")
    cols = ["dataset", "input", "n", "alpha_hill", "kappa_fit", "rmse_fit", "kappa_hill", "rmse_hill", "frac_zero",
            "var_infinite_hill", "mean_infinite_hill"]
    with open(os.path.join(OUT, "fit_table.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in res.iterrows():
            f.write("| " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n")

    # density figure: density of |z - median| from log-spaced bins, on log-log axes, with the fitted density of |X - mu|.
    # Values equal to the median (zero differences) have no place on a log axis and are left out of the bars.
    ncol = 4
    nrow = math.ceil(len(data) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(22, 5 * nrow), squeeze=False)
    for i, (ax, (label, (z, wc, hl, alpha))) in enumerate(zip(axes.ravel(), data.items())):
        a = np.abs(z - np.median(z))
        n = len(a)
        pos = a[a > 0]
        edges = np.geomspace(pos.min(), pos.max() * 1.0001, 61)
        counts, _ = np.histogram(pos, bins=edges)
        dens = counts / (n * np.diff(edges))
        ok = counts > 0
        ax.bar(edges[:-1][ok], dens[ok], width=np.diff(edges)[ok], align="edge", color="tab:blue", alpha=0.8)
        xs = np.geomspace(pos.min(), pos.max(), 400)
        ax.plot(xs, 2 * np.exp(cf.log_pdf(xs, 0.0, wc["sigma"], wc["kappa"])), "r-", lw=2.2)
        sd = float(np.std(z))
        ax.plot(xs, 2 * np.exp(cf.log_pdf(xs, 0.0, sd, 0.0)), "k--", lw=1.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(dens[ok].min() * 0.3, dens[ok].max() * 5)
        ax.set_title(f"{label}  ($\\kappa$ = {wc['kappa']:.2f})")
        ax.text(0.96, 0.95, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="right", va="top", fontsize=20)
        if i % ncol == 0:                       # y label on the first column only
            ax.set_ylabel("density")
        if i + ncol >= len(data):               # x label only where no panel sits below
            ax.set_xlabel(r"$|x - \mathrm{median}|$")
    for ax in axes.ravel()[len(data):]:
        ax.set_visible(False)
    fig.legend(handles=[Line2D([], [], color="tab:blue", lw=8, alpha=0.8, label="Empirical distribution"),
                        Line2D([], [], color="r", lw=2.2, label=r"coupled Gaussian"),
                        Line2D([], [], color="k", ls="--", lw=1.6, label="normal Gaussian (sample std)")],
               loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "Fig_HeavyTail_Distro.png")); plt.close(fig)

    print("Saved to", OUT)


if __name__ == "__main__":
    main()
