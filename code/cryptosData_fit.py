"""Fit the coupled Gaussian of the paper to daily crypto log returns (maximum likelihood).

Density (d = 1, scale sigma, coupling kappa), as in main.tex:
    p(r) = exp_kappa^{-(1+kappa/2)}(t/2) / Z,   t = ((r-mu)/sigma)^2
         = (1 + kappa t/2)^{-(1/kappa + 1/2)} / Z
    Z    = sigma * Gamma(1/kappa)/Gamma(1/kappa + 1/2) * sqrt(2 pi / kappa)
The density tail is |r|^{-(2/kappa+1)}, so P(|r| > x) ~ x^{-2/kappa}: the variance is finite for kappa < 1, the mean for kappa < 2.
The density is normalized in closed form, so the main fit is maximum likelihood (MLE). Two other fits are run for
comparison, each with its own combined figure:
  scorematch  Hyvarinen score matching: picks a much larger kappa and a poor likelihood.
  hill        kappa from the Hill tail index (top HILL_K days of |r - median|), kappa = 2/alpha, then
              mu and sigma by maximum likelihood with kappa fixed.

Reads  <root>/datasets/crypto_data/<ticker>_log_returns.txt (made by cryptosData.py).
Writes <root>/datasets/crypto_data/results_kappa_gaussian_fit/ (tables, tail_check.tsv and one combined log-scale figure per fit).
"""
import math
import os
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "datasets", "crypto_data")
OUT_DIR = os.path.join(INPUT_DIR, "results_kappa_gaussian_fit")
os.makedirs(OUT_DIR, exist_ok=True)

# The download is a snapshot. The last row can be the current, unfinished day, so cut at a closed day.
LAST_DATE = "2026-10-07"
SEED = 42
KAPPA_MAX = 100.0          # upper bound of the search, checked below that no fit sits on it
KAPPA_STARTS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
HIST_BINS = 150
HILL_K = 50                # number of largest |r - median| used by the Hill estimator
TAIL_SDS = (4, 6, 10)      # tail check: days beyond this many sample standard deviations

torch.set_default_dtype(torch.float64)
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "axes.labelsize": 22, "axes.titlesize": 24,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 16, "savefig.dpi": 200,
})


def score_loss(x, mu, sigma, kappa):
    """Mean Hyvarinen score, E[ s^2/2 + s' ], with s = d/dr log p. At kappa = 0 it is the Gaussian score."""
    u = x - mu
    c = 1.0 + kappa / 2.0
    D = 1.0 + kappa * u * u / (2.0 * sigma ** 2)
    s = -c * u / (sigma ** 2 * D)
    ds = -c / (sigma ** 2 * D) + c * kappa * u * u / (sigma ** 4 * D ** 2)
    return torch.mean(0.5 * s * s + ds)


def log_pdf(r, mu, sigma, kappa):
    """Log density of the coupled Gaussian (numpy). kappa = 0 gives the normal Gaussian."""
    t = ((r - mu) / sigma) ** 2
    if kappa < 1e-10:
        return -0.5 * t - 0.5 * math.log(2 * math.pi) - math.log(sigma)
    logZ = (math.log(sigma) + math.lgamma(1 / kappa) - math.lgamma(1 / kappa + 0.5)
            + 0.5 * math.log(2 * math.pi / kappa))
    return -(1 / kappa + 0.5) * np.log1p(kappa * t / 2) - logZ


def fit_kappa_gaussian(x):
    """Minimize the score on standardized data from several starts, return parameters in original units."""
    m, s = float(np.mean(x)), float(np.std(x))
    z = torch.tensor((x - m) / s)
    best = None
    for k0 in KAPPA_STARTS:
        mu = torch.tensor(0.0, requires_grad=True)
        log_sig = torch.tensor(0.0, requires_grad=True)
        k_raw = torch.tensor(math.log((k0 / KAPPA_MAX) / (1 - k0 / KAPPA_MAX)), requires_grad=True)
        opt = torch.optim.LBFGS([mu, log_sig, k_raw], lr=1.0, max_iter=500,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = score_loss(z, mu, torch.exp(log_sig), KAPPA_MAX * torch.sigmoid(k_raw))
            loss.backward()
            return loss

        for _ in range(5):
            opt.step(closure)
        with torch.no_grad():
            kappa = float(KAPPA_MAX * torch.sigmoid(k_raw))
            sig = float(torch.exp(log_sig))
            val = float(score_loss(z, mu, torch.exp(log_sig), torch.tensor(kappa)))
        if best is None or val < best["hyv_std"]:
            best = {"mu_std": float(mu.detach()), "sigma_std": sig, "kappa": kappa, "hyv_std": val}
    return {"mu": m + s * best["mu_std"], "sigma": s * best["sigma_std"], "kappa": best["kappa"],
            "hyv_kappa": best["hyv_std"] / s ** 2}


def fit_ml(x):
    """Maximum likelihood on (mu, log sigma, log kappa), Nelder-Mead from several kappa starts."""
    m, sd = float(np.mean(x)), float(np.std(x))
    nll = lambda p: -np.mean(log_pdf(x, p[0], math.exp(p[1]), math.exp(p[2])))
    runs = [minimize(nll, [m, math.log(sd), math.log(k0)], method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-12, maxiter=5000)) for k0 in KAPPA_STARTS]
    r = min(runs, key=lambda r: r.fun)
    return {"mu": float(r.x[0]), "sigma": math.exp(r.x[1]), "kappa": math.exp(r.x[2]), "ll": -float(r.fun)}


def hill_alpha(x, k=HILL_K):
    """Hill estimate of the tail index alpha (P(|x| > r) ~ r^-alpha) from the k largest |x - median|."""
    a = np.sort(np.abs(x - np.median(x)))[::-1]
    return 1.0 / float(np.mean(np.log(a[:k] / a[k])))


def fit_hill(x):
    """kappa from the Hill tail index (the density tail r^-(2/kappa+1) gives P(|x| > r) ~ r^-(2/kappa), so
    alpha = 2/kappa), then ML for mu and sigma at that kappa."""
    alpha = hill_alpha(x)
    kappa = 2.0 / alpha
    m, sd = float(np.mean(x)), float(np.std(x))
    nll = lambda p: -np.mean(log_pdf(x, p[0], math.exp(p[1]), kappa))
    r = minimize(nll, [m, math.log(sd)], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=5000))
    return {"mu": float(r.x[0]), "sigma": math.exp(r.x[1]), "kappa": kappa, "ll": -float(r.fun), "alpha_hill": alpha}


def expected_beyond(n, r, mu, sigma, kappa):
    """Expected number of days with |x - mu| > r under the fitted density."""
    if kappa < 1e-10:
        return n * math.erfc(r / (sigma * math.sqrt(2)))
    pdf = lambda t: math.exp(log_pdf(np.array(t), mu, sigma, kappa))
    return n * 2 * quad(pdf, mu + r, np.inf, limit=200)[0]


def fit_gaussian(x):
    """Score-matching Gaussian: the minimizer is the sample mean and the (1/N) standard deviation."""
    mu, sigma = float(np.mean(x)), float(np.std(x))
    return {"mu": mu, "sigma": sigma, "hyv_norm": float(score_loss(torch.tensor(x), mu, sigma, 0.0)),
            "ll": float(np.mean(log_pdf(x, mu, sigma, 0.0)))}


def draw_panel(ax, x, fit, norm, asset):
    """Histogram of the returns with the coupled Gaussian fit and the normal Gaussian on a log vertical axis."""
    counts, edges = np.histogram(x, bins=HIST_BINS, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    d = np.abs(centers) / np.max(np.abs(centers))
    colors = np.array([(0.6 - 0.6 * v, 0.8 - 0.8 * v, 1.0 - 1.0 * v) for v in d])
    xs = np.linspace(x.min() - 0.1 * x.std(), x.max() + 0.1 * x.std(), 4000)
    pk = np.exp(log_pdf(xs, fit["mu"], fit["sigma"], fit["kappa"]))
    pn = np.exp(log_pdf(xs, norm["mu"], norm["sigma"], 0.0))
    keep = counts > 0
    ax.bar(centers[keep], counts[keep], width=width, color=colors[keep], alpha=0.9)
    ax.plot(xs, pk, "r-", lw=2.5)
    ax.plot(xs, pn, "k--", lw=1.8)
    ax.set_yscale("log")
    ax.set_ylim(counts[keep].min() * 0.02, max(pk.max(), counts.max()) * 3)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$P(r)$"); ax.set_title(f"{asset}  ($\\kappa$ = {fit['kappa']:.2f})")
    return ax


def legend_handles():
    return [Patch(facecolor=(0.6, 0.8, 1.0), label="Empirical distribution"),
            Line2D([], [], color="r", lw=2.5, label="coupled Gaussian"),
            Line2D([], [], color="k", ls="--", lw=1.8, label="normal Gaussian")]


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith("_log_returns.txt"))
    rows, tails, data = [], [], {}
    for fname in files:
        asset = fname.replace("_log_returns.txt", "")
        df = pd.read_csv(os.path.join(INPUT_DIR, fname), sep="\t")
        df = df[df["Date"] <= LAST_DATE].dropna(subset=["LogReturn"])
        x = df["LogReturn"].values
        n = len(x)
        norm = fit_gaussian(x)
        fits = {"mle": fit_ml(x), "scorematch": fit_kappa_gaussian(x), "hill": fit_hill(x)}
        for f in fits.values():
            f["hyv"] = float(score_loss(torch.tensor(x), f["mu"], f["sigma"], f["kappa"]))
            f["ll"] = float(np.mean(log_pdf(x, f["mu"], f["sigma"], f["kappa"])))
        data[asset] = (x, fits, norm)
        row = {"Asset": asset, "Start": df["Date"].iloc[0], "End": df["Date"].iloc[-1], "N": n,
               "ll_normal": norm["ll"], "hyv_normal": norm["hyv_norm"], "alpha_hill": hill_alpha(x)}
        for name, f in fits.items():
            row.update({f"mu_{name}": f["mu"], f"sigma_{name}": f["sigma"], f"kappa_{name}": f["kappa"],
                        f"tail_index_{name}": 2 / f["kappa"], f"ll_{name}": f["ll"], f"hyv_{name}": f["hyv"]})
            if f["kappa"] > 0.99 * KAPPA_MAX:
                print(f"  WARNING: {asset} {name} kappa sits on the search bound {KAPPA_MAX}")
        rows.append(row)
        sd = float(np.std(x))
        for k in TAIL_SDS:
            r = k * sd
            t = {"Asset": asset, "k_sd": k, "observed": int(np.sum(np.abs(x - np.median(x)) > r)),
                 "gaussian": expected_beyond(n, r, norm["mu"], norm["sigma"], 0.0)}
            for name, f in fits.items():
                t[name] = expected_beyond(n, r, f["mu"], f["sigma"], f["kappa"])
            tails.append(t)
        print(f"{asset}: N={n} Hill alpha={row['alpha_hill']:.2f} | " + " | ".join(
            f"{nm}: kappa={f['kappa']:.3f} ll={f['ll']:.3f}" for nm, f in fits.items()) + f" | normal ll={norm['ll']:.3f}")
        # sanity: the MLE density integrates to 1
        f = fits["mle"]
        xs = np.linspace(-200 * f["sigma"], 200 * f["sigma"], 400001)
        print(f"  mass within +-200 sigma (mle): {np.trapezoid(np.exp(log_pdf(xs, f['mu'], f['sigma'], f['kappa'])), xs):.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fit_results.tsv"), sep="\t", index=False)
    pd.DataFrame(tails).to_csv(os.path.join(OUT_DIR, "tail_check.tsv"), sep="\t", index=False, float_format="%.2f")

    names = {"mle": "Fig_Crypto_Distro_Combined.png", "scorematch": "Fig_Crypto_Distro_Combined_scorematch.png",
             "hill": "Fig_Crypto_Distro_Combined_hill.png"}
    for name, fname in names.items():
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        for ax, (asset, (x, fits, norm)) in zip(axes.ravel(), data.items()):
            draw_panel(ax, x, fits[name], norm, asset)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT_DIR, fname)); plt.close(fig)

    with open(os.path.join(OUT_DIR, "table_rows.tex"), "w") as f:
        for r in rows:
            f.write(f"{r['Asset']} & {r['Start']} & {r['End']} &\n"
                    f"${r['mu_mle']:.2e}$ & {r['sigma_mle']:.5f} & {r['kappa_mle']:.3f} & "
                    f"${r['ll_mle']:.3f}$ & ${r['ll_normal']:.3f}$ & "
                    f"${r['hyv_mle']:.1f}$ & ${r['hyv_normal']:.1f}$ \\\\\n\n")
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
