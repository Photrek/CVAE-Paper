"""Pilot: do real datasets have a tail index below 2 (infinite variance)?

Run with  python3 -I code/heavytail_pilot.py  (the folders in datasets/ are untrusted downloads, made by heavytail_download.py).

For each candidate series it builds
  raw   the positive one-sided series x_t (Hill index of the k largest values),
  diff  the signed difference x_t - x_(t-1) of consecutive records,
  logr  the signed log-ratio log(x_t / x_(t-1)) (only where both values are positive),
and for the signed series it reports the Hill alpha of |x - median| (the function of cryptosData_fit.py) at several k
and the maximum-likelihood coupled-Gaussian kappa in d = 1 (fit_ml of cryptosData_fit.py). The coupled Gaussian has
P(|x| > t) ~ t^(-2/kappa), so alpha = 2/kappa. The fit is on data divided by the median absolute deviation, because kappa
does not depend on the scale and the sample standard deviation is unstable when the variance is infinite.

Writes heavytail_results/screening/ at the project root (gitignored): results.tsv, hill_vs_k.tsv, results_table.md and
tail_plots_{raw,diff,logr}.png. No model is trained.
"""
import glob
import warnings
import gzip
import importlib.util
import io
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

warnings.filterwarnings("ignore")   # overflow in log_pdf during Nelder-Mead steps with extreme kappa
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "heavytail_results", "screening")

# Reuse the crypto fit code (hill_alpha, fit_ml, log_pdf). `-I` removes the script folder from sys.path, so load by path.
_spec = importlib.util.spec_from_file_location("cryptosData_fit", os.path.join(HERE, "cryptosData_fit.py"))
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

KS = (20, 50, 100, 200, 500, 1000)
MAX_FIT = 200_000               # cap on points for the ML fit (millions of trades or outages)
MIN_N = 3000                    # "a few thousand points"


# ---------------------------------------------------------------- loaders: each returns a time-ordered 1D array (+ gap mask)
def load_quake():
    fs = sorted(glob.glob(os.path.join(DATA, "usgs_quake_data", "quakes_*.csv")))
    df = pd.concat([pd.read_csv(f, usecols=["time", "mag"]) for f in fs]).dropna()
    df = df[df["mag"] >= 5.0].sort_values("time")
    return 10 ** (1.5 * df["mag"].values), None       # radiated energy up to a constant: log10 E = 1.5 M + 4.8


def load_danish():
    df = pd.read_csv(os.path.join(DATA, "danish_fire_data", "danish.csv"))
    return df["dat"].values.astype(float), None       # losses in millions of DKK, in time order (1980 to 1990)


def load_goes():
    rows = []
    pat = re.compile(r"^\s*(\d{5})(\d{6})\s+(\d{4})\s.*?\s([ABCMX])\s+(\d+)\s+GOES")
    for f in sorted(glob.glob(os.path.join(DATA, "goes_flare_data", "goes-xrs-report_*.txt"))):
        for line in open(f, errors="replace"):
            m = pat.match(line)
            if m:
                yymmdd = m.group(2)
                yy = int(yymmdd[:2])
                key = (1900 + yy if yy >= 70 else 2000 + yy) * 10 ** 8 + int(yymmdd[2:]) * 10 ** 4 + int(m.group(3))
                peak = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[m.group(4)] * int(m.group(5)) / 10
                rows.append((key, peak))
    rows.sort()
    return np.array([r[1] for r in rows]), None       # peak X-ray flux in W/m^2


def load_nwis(site):
    f = os.path.join(DATA, "nwis_river_data", f"{site}_discharge.rdb")
    df = pd.read_csv(f, sep="\t", comment="#", skiprows=[1])
    qcol = [c for c in df.columns if c.endswith("_00060_00003")][0]    # daily mean discharge; some sites list several series
    df = df.rename(columns={"datetime": "date", qcol: "q"})
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df.dropna(subset=["q"])
    d = pd.to_datetime(df["date"])
    ok = (d.diff().dt.days == 1).values                # True where the previous record is the previous day
    return df["q"].values.astype(float), ok


def load_ghcn(station):
    raw = gzip.open(os.path.join(DATA, "ghcn_rain_data", f"{station}.csv.gz"), "rt").read()
    df = pd.read_csv(io.StringIO(raw), header=None, usecols=[1, 2, 3, 5],
                     names=["date", "elem", "val", "qflag"], dtype={"date": str, "qflag": str})
    df = df[(df["elem"] == "PRCP") & df["qflag"].isna()].sort_values("date")
    d = pd.to_datetime(df["date"], format="%Y%m%d")
    ok = (d.diff().dt.days == 1).values
    return df["val"].values.astype(float) / 10.0, ok   # mm


def load_crypto_hourly(sym):
    fs = sorted(glob.glob(os.path.join(DATA, "crypto_hourly_data", f"{sym}-1h-*.csv")))
    t, c = [], []
    for f in fs:
        a = np.loadtxt(f, delimiter=",", usecols=(0, 4))
        t.append(a[:, 0]); c.append(a[:, 1])
    t, c = np.concatenate(t), np.concatenate(c)
    t = np.where(t > 1e14, t / 1000, t)                # 2025 files use microseconds
    ok = np.r_[False, np.diff(t) == 3600 * 1000]
    return c, ok


# ---------------------------------------------------------------- round 2 loaders
def load_csv_col(folder, fname, col):
    return pd.read_csv(os.path.join(DATA, folder, fname))[col].values.astype(float), None   # file order = time order


def _usd(s):
    """Storm Events damage strings: '10.00K', '1.5M', '2B', '0'."""
    s = str(s).strip().upper()
    if s in ("", "NAN"):
        return np.nan
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1], None)
    try:
        return float(s[:-1]) * mult if mult else float(s)
    except ValueError:
        return np.nan


def load_storm(col="DAMAGE_PROPERTY"):
    fs = sorted(glob.glob(os.path.join(DATA, "noaa_storm_data", "StormEvents_details-ftp_v1.0_d*.csv.gz")))
    fs = [f for f in fs if int(re.search(r"_d(\d{4})", f).group(1)) >= 1996]    # all event types are recorded from 1996
    df = pd.concat([pd.read_csv(f, usecols=["BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", col], low_memory=False) for f in fs])
    df["usd"] = df[col].map(_usd)
    df = df[df["usd"] > 0].sort_values(["BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME"], kind="stable")
    return df["usd"].values, None                     # US dollars per event, nominal


def load_wiki(article):
    import json
    items = json.load(open(os.path.join(DATA, "wikipedia_views_data", f"{article}.json")))["items"]
    t = pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d")
    return np.array([i["views"] for i in items], dtype=float), np.r_[False, np.diff(t.values).astype("timedelta64[D]").astype(int) == 1]


def load_binance_trades():
    fs = sorted(glob.glob(os.path.join(DATA, "binance_trades_data", "BTCUSDT-aggTrades-*.csv")))
    q = np.concatenate([pd.read_csv(f, header=None, usecols=[2]).values.ravel() for f in fs]).astype(float)
    return q, None                                    # trade size in BTC, in trade order (days joined)


def load_hn(col="points"):
    df = pd.read_csv(os.path.join(DATA, "hackernews_data", "hn_stories.tsv"), sep="\t").sort_values("t", kind="stable")
    return df[col].values.astype(float), None


def load_indegree(fname):
    e = pd.read_csv(os.path.join(DATA, "snap_graph_data", fname), sep=r"\s+", comment="#", header=None, names=["a", "b"])
    return np.bincount(e["b"].values).astype(float), None  # in-degree per node id


def load_eaglei(kind):
    fs = sorted(glob.glob(os.path.join(DATA, "eaglei_outage_data", "eaglei_outages_*_merged.csv")))
    df = pd.concat([pd.read_csv(f, usecols=["start_time", "duration", "max_customers", "mean_customers"]) for f in fs])
    df = df.sort_values("start_time", kind="stable")
    x = df["duration"] * df["mean_customers"] if kind == "customer_hours" else df["max_customers"]
    return x.values.astype(float), None               # county outages, ordered by start time (ties in file order)


def load_wildfire(min_year=1950):
    f = glob.glob(os.path.join(DATA, "wildfire_area_data", "NFDB_point_*.txt"))[0]
    df = pd.read_csv(f, encoding="latin1", usecols=["YEAR", "MONTH", "DAY", "SIZE_HA"], low_memory=False)
    df = df[(df["SIZE_HA"] > 0) & (df["YEAR"] >= min_year)].sort_values(["YEAR", "MONTH", "DAY"], kind="stable")
    return df["SIZE_HA"].values.astype(float), None   # hectares per fire, Canada, in date order


def load_gcmt(min_mw=0.0):
    """Global CMT ndk files: 5 lines per event. Line 1 has the date, line 4 starts with the exponent and the 11th field
    of line 5 is the scalar-moment mantissa (3 significant digits). Returns M0 in dyne-cm, in time order."""
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "gcmt_quake_data", "*.ndk"))):
        lines = open(f, errors="replace").read().splitlines()
        for i in range(0, len(lines) - 4, 5):
            try:
                m0 = float(lines[i + 4].split()[10]) * 10.0 ** int(lines[i + 3][:2])
            except (ValueError, IndexError):
                continue
            rows.append((lines[i][5:26], m0))
    rows = sorted(set(rows))                           # the monthly files do not overlap the 1976-2020 file, but be safe
    m0 = np.array([r[1] for r in rows])
    mw = 2 / 3 * np.log10(m0) - 10.7
    return m0[mw >= min_mw], None


def load_us_wildfire(min_acres=0.0):
    """FPA FOD 7th edition as counts per (year, size): expand to one value per fire, year order (order inside a year lost)."""
    df = pd.read_csv(os.path.join(DATA, "us_wildfire_data", "fpa_fod7_year_size_count.csv"))
    df = df[(df["fire_size_acres"] > 0) & (df["fire_size_acres"] >= min_acres)].sort_values(["year", "fire_size_acres"])
    return np.repeat(df["fire_size_acres"].values.astype(float), df["count"].values), None


def load_oe417(what):
    """DOE OE-417 events from the yearly xls files (layouts differ, so find the header row by its text). File ids
    grow with the year; `what` is a substring of the column header ('Customers Affected' or 'Loss')."""
    fs = sorted(glob.glob(os.path.join(DATA, "blackout_data", "oe417_*.xls")), key=lambda f: int(re.search(r"_(\d+)\.xls", f).group(1)))
    vals = []
    for f in fs:
        x = pd.read_excel(f, header=None)
        h = [i for i in range(min(8, len(x))) if x.iloc[i].astype(str).str.contains("Customers Affected", case=False).any()][0]
        c = [j for j, v in enumerate(x.iloc[h].astype(str)) if what.lower() in v.lower()][0]
        v = pd.to_numeric(x.iloc[h + 1:, c].astype(str).str.replace(",", "").str.extract(r"([\d.]+)")[0], errors="coerce")
        vals.append(v.dropna().values)
    v = np.concatenate(vals)
    return v[v > 0], None


def load_breach():
    """HHS OCR health-data breaches (open cases and archive): people affected, in submission-date order."""
    df = pd.concat([pd.read_csv(os.path.join(DATA, "hhs_breach_data", f"hhs_breach_{n}.csv")) for n in ("open", "archive")])
    df = df.drop_duplicates()
    df["t"] = pd.to_datetime(df["Breach Submission Date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("t", kind="stable")
    return pd.to_numeric(df["Individuals Affected"], errors="coerce").values.astype(float), None


def load_citations(year=2000):
    """OpenAlex articles of one year as (cites, count): expand to one value per article (no time order)."""
    df = pd.read_csv(os.path.join(DATA, "openalex_citation_data", f"openalex_articles_{year}_cites.csv"))
    return np.repeat(df["cites"].values.astype(float), df["count"].values), None


def load_words():
    """Number of times each distinct word appears in Moby Dick (sorted, so no order)."""
    return np.loadtxt(os.path.join(DATA, "word_freq_data", "mobydick_word_counts.txt")), None


CANDIDATES = {
    "quake_energy": load_quake,
    "danish_fire": load_danish,
    "goes_flare_peak": load_goes,
    "nwis_09484000_Sabino": lambda: load_nwis("09484000"),
    "nwis_09471000_SanPedro": lambda: load_nwis("09471000"),
    "nwis_08158000_ColoradoTX": lambda: load_nwis("08158000"),
    "nwis_07010000_Mississippi": lambda: load_nwis("07010000"),
    "nwis_01646500_Potomac": lambda: load_nwis("01646500"),
    "ghcn_NYC_rain": lambda: load_ghcn("USW00094728"),
    "crypto_BTC_1h": lambda: load_crypto_hourly("BTCUSDT"),
    "crypto_ETH_1h": lambda: load_crypto_hourly("ETHUSDT"),
}
# round 2 (2026-10-08)
CANDIDATES.update({
    "norwegian_fire": lambda: load_csv_col("norwegian_fire_data", "norwegianfire.csv", "size"),
    "soa_medical": lambda: load_csv_col("norwegian_fire_data", "soa.csv", "size"),
    "storm_property_damage": load_storm,
    "storm_crop_damage": lambda: load_storm("DAMAGE_CROPS"),
    "wildfire_area_Canada": load_wildfire,
    "oe417_customers": lambda: load_oe417("Customers Affected"),
    "oe417_demand_loss_MW": lambda: load_oe417("Loss"),
    "binance_BTC_trade_size": load_binance_trades,
    "hackernews_points": load_hn,
    "hackernews_comments": lambda: load_hn("comments"),
    "web_Stanford_indegree": lambda: load_indegree("web-Stanford.txt.gz"),
    "Epinions_indegree": lambda: load_indegree("soc-Epinions1.txt.gz"),
    "eaglei_customer_hours": lambda: load_eaglei("customer_hours"),
    "eaglei_max_customers": lambda: load_eaglei("max_customers"),
})
# round 3 (2026-10-13): replacements for Canada wildfire, Sabino creek and the rounded USGS magnitudes
CANDIDATES.update({
    "gcmt_moment": load_gcmt,
    "gcmt_moment_mw55": lambda: load_gcmt(5.5),
    "us_wildfire_area": load_us_wildfire,
    "us_wildfire_area_10ac": lambda: load_us_wildfire(10.0),
    "us_wildfire_area_100ac": lambda: load_us_wildfire(100.0),
    "nwis_11477000_Eel": lambda: load_nwis("11477000"),
    "nwis_11467000_Russian": lambda: load_nwis("11467000"),
    "nwis_08176500_GuadalupeVictoria": lambda: load_nwis("08176500"),
    "nwis_08167500_GuadalupeSpringBranch": lambda: load_nwis("08167500"),
    "nwis_02489500_Pearl": lambda: load_nwis("02489500"),
})
for _a in ("Bitcoin", "Donald_Trump", "Elon_Musk", "COVID-19_pandemic", "Taylor_Swift", "Game_of_Thrones", "Wikipedia",
           "Barack_Obama", "Facebook", "Cristiano_Ronaldo", "ChatGPT", "Michael_Jackson"):
    CANDIDATES["wiki_" + _a] = (lambda a=_a: load_wiki(a))
for _s, _n in (("USW00023183", "Phoenix"), ("USW00023160", "Tucson"), ("USW00023195", "Yuma"), ("USW00023169", "LasVegas"),
               ("USC00042319", "DeathValley"), ("USW00012918", "HoustonHobby"), ("USW00012839", "Miami"),
               ("USW00012960", "HoustonIAH")):
    CANDIDATES["ghcn_" + _n + "_rain"] = (lambda s=_s: load_ghcn(s))
# round 4 (2026-10-13): new kinds of data for the 8th panel
CANDIDATES.update({
    "hhs_breach_individuals": load_breach,
    "openalex_cites_2000": load_citations,
    "mobydick_word_freq": load_words,
})
RAW_ONLY = {"web_Stanford_indegree", "Epinions_indegree",       # node order has no time meaning, so no differences
            "norwegian_fire", "soa_medical",                     # claims are sorted by size within a year (no order inside the year)
            "openalex_cites_2000", "mobydick_word_freq"}         # sorted counts, no order
PRICE_LIKE = {"crypto_BTC_1h", "crypto_ETH_1h"}        # the level is a price, so only the log return makes sense


# ---------------------------------------------------------------- estimators
def hill_one_sided(x, k):
    a = np.sort(x[x > 0])[::-1]
    if len(a) <= k or a[k] <= 0:
        return float("nan")
    return 1.0 / float(np.mean(np.log(a[:k] / a[k])))


def hill_signed(x, k):
    a = np.sort(np.abs(x - np.median(x)))[::-1]
    if len(a) <= k or a[k] <= 0:
        return float("nan")
    return cf.hill_alpha(x, k)


def survival_fit(t, kappa, sigma):
    """P(|X - mu| > t) of the fitted coupled Gaussian (mu = 0), by numerical integration."""
    f = lambda u: math.exp(cf.log_pdf(np.array(u), 0.0, sigma, kappa))
    return 2 * quad(f, t, np.inf, limit=200)[0]


def build(name):
    x, ok = CANDIDATES[name]()
    out = {}
    if name not in PRICE_LIKE:
        out["raw"] = x
    if ok is None:
        ok = np.ones(len(x), dtype=bool); ok[0] = False
    if name in RAW_ONLY:
        return out
    ok = ok[1:]                                         # pairs (x_(t-1), x_t) with consecutive records
    a, b = x[:-1][ok], x[1:][ok]
    if name not in PRICE_LIKE:
        out["diff"] = b - a
    pos = (a > 0) & (b > 0)
    out["logr"] = np.log(b[pos] / a[pos])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(cf.SEED)
    rows, hill_rows, series = [], [], {}
    for name in CANDIDATES:
        try:
            parts = build(name)
        except Exception as e:
            print(f"{name}: not loaded ({type(e).__name__}: {e})")
            continue
        for kind, x in parts.items():
            x = x[np.isfinite(x)]
            n = len(x)
            row = {"dataset": name, "series": kind, "n": n}
            hills = {k: (hill_one_sided(x, k) if kind == "raw" else hill_signed(x, k)) for k in KS if k <= n // 10}
            for k, v in hills.items():
                hill_rows.append({"dataset": name, "series": kind, "k": k, "alpha": v})
                row[f"alpha_k{k}"] = v
            vals = [v for v in hills.values() if np.isfinite(v)]
            row["alpha_min"], row["alpha_max"] = (min(vals), max(vals)) if vals else (np.nan, np.nan)
            if kind != "raw":
                mad = float(np.median(np.abs(x - np.median(x))))
                z = (x - np.median(x)) / mad if mad > 0 else None
                if z is not None and len(z) > MAX_FIT:           # Hill uses every point, the ML fit a fixed random subset
                    z = z[np.random.default_rng(cf.SEED).choice(len(z), MAX_FIT, replace=False)]
                if z is not None:
                    fit = cf.fit_ml(z)
                    row.update(kappa_ml=fit["kappa"], alpha_ml=2 / fit["kappa"], sigma_ml=fit["sigma"], mu_ml=fit["mu"])
                    h200 = hills.get(200, np.nan)
                    row["kappa_hill200"] = 2 / h200 if np.isfinite(h200) else np.nan
                    series[(name, kind)] = (z, fit, hills)
                else:
                    print(f"  {name}/{kind}: MAD is 0, no fit")
            else:
                series[(name, kind)] = (x[x > 0], None, hills)
            rows.append(row)
            print(f"{name:28s} {kind:5s} n={n:7d} alpha(k=50..)={[round(v, 2) for v in hills.values()]}"
                  + (f" kappa_ml={row['kappa_ml']:.3f} (2/kappa={row['alpha_ml']:.2f})" if "kappa_ml" in row else ""))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.tsv"), sep="\t", index=False, float_format="%.4g")
    pd.DataFrame(hill_rows).to_csv(os.path.join(OUT, "hill_vs_k.tsv"), sep="\t", index=False, float_format="%.4g")

    # markdown table
    cols = ["dataset", "series", "n"] + [f"alpha_k{k}" for k in KS] + ["kappa_ml", "alpha_ml", "kappa_hill200"]
    with open(os.path.join(OUT, "results_table.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in res.iterrows():
            f.write("| " + " | ".join("" if (c not in r or (isinstance(r[c], float) and np.isnan(r[c])))
                                      else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |\n")

    # tail plots, one figure per series kind
    for kind in ("raw", "diff", "logr"):
        keys = [k for k in series if k[1] == kind]
        if not keys:
            continue
        nc = 3
        nr = math.ceil(len(keys) / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 4.6 * nr), squeeze=False)
        for ax, key in zip(axes.ravel(), keys):
            z, fit, hills = series[key]
            a = np.sort(np.abs(z) if kind != "raw" else z)
            n = len(a)
            sf = 1 - np.arange(1, n + 1) / (n + 1)       # P(A > a_i)
            keep = a > 0
            ax.loglog(a[keep], sf[keep], ".", ms=3, color="tab:blue", label="data")
            k0 = 200 if 200 in hills else max(hills, default=None)
            al = hills[k0] if k0 else np.nan
            if np.isfinite(al):
                t0 = np.sort(a)[::-1][k0]
                tt = np.geomspace(t0, a.max(), 30)
                ax.loglog(tt, (k0 / n) * (tt / t0) ** (-al), "k:", lw=1.6, label=f"Hill $\\alpha$={al:.2f} (k={k0})")
            if fit is not None and fit["kappa"] < 20:       # a larger kappa is a degenerate fit, not drawn
                tt = np.geomspace(max(a[keep].min(), 1e-3), a.max(), 25)
                ax.loglog(tt, [survival_fit(t, fit["kappa"], fit["sigma"]) for t in tt], "r-", lw=1.8,
                          label=f"ML coupled, $\\kappa$={fit['kappa']:.2f}")
            ax.set_ylim(max(1.0 / n * 0.3, 1e-8), 1.5)
            ax.set_title(f"{key[0]}  (n={n})", fontsize=11)
            ax.set_xlabel("|x - median| / MAD" if kind != "raw" else "x")
            ax.set_ylabel("P(X > t)")
            ax.legend(fontsize=8)
        for ax in axes.ravel()[len(keys):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"tail_plots_{kind}.png"), dpi=130); plt.close(fig)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
