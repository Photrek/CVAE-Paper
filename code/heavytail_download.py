"""Download the candidate infinite-variance datasets, each into its own gitignored folder in datasets/.

    python3 heavytail_download.py [name ...]      names: see JOBS at the bottom (default: all)

Only downloads. The analysis (heavytail_pilot.py) reads the files with `python3 -I`.
Folders: usgs_quake_data/, danish_fire_data/, goes_flare_data/, nwis_river_data/, ghcn_rain_data/, crypto_hourly_data/.
Round 2: norwegian_fire_data/, noaa_storm_data/, wikipedia_views_data/, wildfire_area_data/, binance_trades_data/,
hackernews_data/, snap_graph_data/, eaglei_outage_data/, blackout_data/. `ghcn_flash` adds stations to ghcn_rain_data/.
Round 3 (replacements, 2026-10-13): gcmt_quake_data/, us_wildfire_data/; `nwis_perennial` adds rivers to nwis_river_data/.
Round 4 (new kinds of data, 2026-10-13): hhs_breach_data/, openalex_citation_data/, word_freq_data/.
"""
import io
import os
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "cvae-paper-pilot/1.0"}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as e:  # network errors: wait and retry
            print(f"  retry {i + 1} for {url[:90]}: {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError("download failed: " + url)


def out_dir(name):
    d = os.path.join(ROOT, "datasets", name)
    os.makedirs(d, exist_ok=True)
    return d


def quake():
    """USGS ComCat, M >= 5.0, one CSV per year (the API caps a query at 20000 events)."""
    d = out_dir("usgs_quake_data")
    for y in range(1973, 2027):
        f = os.path.join(d, f"quakes_{y}.csv")
        if os.path.exists(f) and y < 2026:
            continue
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minmagnitude=5&orderby=time-asc"
               f"&eventtype=earthquake&starttime={y}-01-01&endtime={y + 1}-01-01")
        open(f, "wb").write(get(url))
        print("quake", y)


def danish():
    """Danish fire insurance claims (Copenhagen Re, 2167 losses over 1 million DKK, 1980-1990), via Rdatasets."""
    d = out_dir("danish_fire_data")
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/evir/danish.csv"
    open(os.path.join(d, "danish.csv"), "wb").write(get(url))


def goes():
    """NOAA NGDC GOES X-ray flare reports, 1975 to now, one text file per year."""
    d = out_dir("goes_flare_data")
    base = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"
    for y in range(1975, 2027):
        f = os.path.join(d, f"goes-xrs-report_{y}.txt")
        try:
            data = get(base + f"goes-xrs-report_{y}.txt", tries=2)
            open(f, "wb").write(data)
            print("goes", y)
        except RuntimeError:
            print("goes: no file for", y)


NWIS_SITES = {  # flashy, unregulated rivers (peaks far above the typical flow) and two large ones for contrast
    "09484000": "Sabino Creek near Tucson AZ",
    "09471000": "San Pedro River at Charleston AZ",
    "08158000": "Colorado River at Austin TX",
    "07010000": "Mississippi River at St. Louis MO",
    "01646500": "Potomac River near Washington DC",
}


def nwis():
    """USGS NWIS daily mean discharge (00060, ft3/s), full record."""
    d = out_dir("nwis_river_data")
    for site in NWIS_SITES:
        url = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd=00060"
               "&startDT=1900-01-01&endDT=2026-10-01")
        open(os.path.join(d, f"{site}_discharge.rdb"), "wb").write(get(url))
        print("nwis", site)


GHCN_STATIONS = ["USW00094728"]  # New York Central Park, daily record from 1869


def ghcn():
    """NOAA GHCN-Daily, whole station file (csv.gz from the by_station folder)."""
    d = out_dir("ghcn_rain_data")
    for s in GHCN_STATIONS:
        url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{s}.csv.gz"
        open(os.path.join(d, f"{s}.csv.gz"), "wb").write(get(url))
        print("ghcn", s)


def crypto_hourly():
    """Binance public monthly 1h klines (data.binance.vision), BTCUSDT and ETHUSDT, 2019-01 to 2026-09."""
    d = out_dir("crypto_hourly_data")
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for y in range(2019, 2027):
            for m in range(1, 13):
                if (y, m) > (2026, 9):
                    break
                f = os.path.join(d, f"{sym}-1h-{y}-{m:02d}.csv")
                if os.path.exists(f):
                    continue
                url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip"
                z = zipfile.ZipFile(io.BytesIO(get(url)))
                open(f, "wb").write(z.read(z.namelist()[0]))
        print("crypto_hourly", sym)


def norwegian():
    """Norwegian fire claims (9181, 1972-1992, 1000 NOK, priority 500) and SOA group medical claims (75789, 1991),
    from the .RData files of the CRAN package ReIns, converted to CSV with pyreadr (pip install pyreadr)."""
    import pyreadr
    import tarfile
    d = out_dir("norwegian_fire_data")
    tf = tarfile.open(fileobj=io.BytesIO(get("https://cran.r-project.org/src/contrib/ReIns_1.0.16.tar.gz")))
    for name in ("norwegianfire", "soa"):
        raw = tf.extractfile(f"ReIns/data/{name}.RData").read()
        tmp = os.path.join(d, f"{name}.RData")
        open(tmp, "wb").write(raw)
        df = list(pyreadr.read_r(tmp).values())[0]
        df.to_csv(os.path.join(d, f"{name}.csv"), index=False)
        print("norwegian", name, df.shape)


def storm():
    """NOAA Storm Events details (property and crop damage, 1950 to now), one csv.gz per year."""
    import re
    d = out_dir("noaa_storm_data")
    base = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    names = sorted(set(re.findall(r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz", get(base).decode())))
    for n in names:
        f = os.path.join(d, n[:len("StormEvents_details-ftp_v1.0_d1950")] + ".csv.gz")
        if not os.path.exists(f):
            open(f, "wb").write(get(base + n))
    print("storm", len(names), "years")


WIKI_ARTICLES = ["Bitcoin", "Donald_Trump", "Elon_Musk", "COVID-19_pandemic", "Taylor_Swift", "Game_of_Thrones",
                 "Wikipedia", "Barack_Obama", "Facebook", "Cristiano_Ronaldo", "ChatGPT", "Michael_Jackson"]


def wiki():
    """Wikimedia daily page views (users, all access) of a few spiky English-Wikipedia articles, 2015-07 to 2026-09."""
    import json
    d = out_dir("wikipedia_views_data")
    for a in WIKI_ARTICLES:
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               f"{a}/daily/20150701/20260930")
        f = os.path.join(d, f"{a}.json")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            continue
        data = get(url, tries=6)          # fetch first, so a failed request leaves no empty file
        open(f, "wb").write(data)
        print("wiki", a)
        time.sleep(2)                     # the API answers 429 to fast request bursts


def wildfire():
    """Canadian National Fire Database, point file as text (about 400k fires, SIZE_HA, from 1917)."""
    d = out_dir("wildfire_area_data")
    z = zipfile.ZipFile(io.BytesIO(get("https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_txt.zip")))
    z.extractall(d)
    print("wildfire", z.namelist())


GHCN_FLASH = {  # flash-flood or very wet stations, besides NYC
    "USW00023183": "Phoenix AZ", "USW00023160": "Tucson AZ", "USW00023195": "Yuma AZ", "USW00023169": "Las Vegas NV",
    "USC00042319": "Death Valley CA", "USW00012918": "Houston Hobby TX", "USW00012839": "Miami FL",
    "USW00012960": "Houston Intercontinental TX",
}


def ghcn_flash():
    d = out_dir("ghcn_rain_data")
    for s in GHCN_FLASH:
        try:
            open(os.path.join(d, f"{s}.csv.gz"), "wb").write(get(f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{s}.csv.gz", tries=2))
            print("ghcn_flash", s, GHCN_FLASH[s])
        except RuntimeError:
            print("ghcn_flash: no file for", s)


def binance_trades():
    """Binance public daily aggTrades (individual trade sizes, in time order). Three days of BTCUSDT."""
    d = out_dir("binance_trades_data")
    for day in ["2024-03-12", "2024-08-05", "2025-01-15"]:
        f = os.path.join(d, f"BTCUSDT-aggTrades-{day}.csv")
        if os.path.exists(f):
            continue
        z = zipfile.ZipFile(io.BytesIO(get(f"https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{day}.zip")))
        open(f, "wb").write(z.read(z.namelist()[0]))
        print("binance_trades", day)


def hackernews():
    """Hacker News stories 2018 to 2024 (points, comments) from the Algolia API, one query per day (1000 hits cap)."""
    import datetime
    import json
    d = out_dir("hackernews_data")
    f = os.path.join(d, "hn_stories.tsv")
    t = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    rows = []
    while t < end:
        a, b = int(t.timestamp()), int((t + datetime.timedelta(days=1)).timestamp())
        url = (f"https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=1000&numericFilters=created_at_i%3E%3D{a},created_at_i%3C{b}")
        for h in json.loads(get(url))["hits"]:
            rows.append(f"{h['created_at_i']}\t{h.get('points') or 0}\t{h.get('num_comments') or 0}")
        t += datetime.timedelta(days=1)
        if t.day == 1:
            print("hackernews", t.date())
    open(f, "w").write("t\tpoints\tcomments\n" + "\n".join(sorted(rows)) + "\n")


def snap():
    """SNAP link graphs (edge lists): Stanford web graph and Epinions trust network. In-degree is the heavy variable."""
    d = out_dir("snap_graph_data")
    for n in ("web-Stanford.txt.gz", "soc-Epinions1.txt.gz"):
        open(os.path.join(d, n), "wb").write(get("https://snap.stanford.edu/data/" + n))
        print("snap", n)


def blackout():
    """DOE OE-417 annual summaries (xls). oe.netl.doe.gov does not resolve from here, so use the Wayback copy.
    Files differ by year (headers, units), so heavytail_pilot.py parses them by header text."""
    d = out_dir("blackout_data")
    for i in [84, 82, 81, 80, 79, 78, 77, 76, 75, 62, 41, 31, 22, 11, 15, 14, 10, 9, 8, 7, 6, 5, 4, 3]:
        f = os.path.join(d, f"oe417_{i}.xls")
        if os.path.exists(f):
            continue
        try:
            data = get(f"http://web.archive.org/web/2024id_/https://www.oe.netl.doe.gov/download.aspx?type=OE417XLS&ID={i}", tries=5)
        except RuntimeError:
            print("blackout: no file for", i); continue
        if data[:4] == b"\xd0\xcf\x11\xe0" or data[:2] == b"PK":
            open(f, "wb").write(data)
            print("blackout", i)


def eaglei():
    """EAGLE-I county-level power outages 2014-2023 (DOE/ORNL, via the PNNL/NREL OpenEI submission 6458): one row per
    outage with start_time, duration (h) and customer counts. Keeps the *_merged.csv files only."""
    d = out_dir("eaglei_outage_data")
    z = zipfile.ZipFile(io.BytesIO(get("https://data.openei.org/files/6458/Outage_Dataset_R1.zip")))
    for n in z.namelist():
        if n.endswith("_merged.csv"):
            open(os.path.join(d, os.path.basename(n)), "wb").write(z.read(n))
    print("eaglei", len(os.listdir(d)), "files")


BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}


def gcmt():
    """Global CMT catalog (ndk): scalar seismic moment, 3 significant digits, 1976 to now. ldeo.columbia.edu answers
    403 to unknown user agents. 1976-2020 is one file; later months are separate files."""
    d = out_dir("gcmt_quake_data")
    base = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/"
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    files = [("jan76_dec20.ndk", base + "jan76_dec20.ndk")]
    files += [(f"{m}{y % 100:02d}.ndk", f"{base}NEW_MONTHLY/{y}/{m}{y % 100:02d}.ndk") for y in range(2021, 2027) for m in months]
    for name, url in files:
        f = os.path.join(d, name)
        if os.path.exists(f):
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=BROWSER_UA), timeout=120) as r:
                data = r.read()
        except Exception as e:            # months not yet published give 404
            print("gcmt: no file", name, e); continue
        open(f, "wb").write(data)
        print("gcmt", name)


def us_wildfire():
    """FPA FOD 7th edition (Short 2026, US wildfires 1992-2024, about 2.66M fires) from the USDA ArcGIS service.
    The archive zip is blocked here, so this asks the service for the count of fires per (year, fire_size in acres).
    Grouped queries return at most 2000 rows and ignore resultOffset, so a size range that hits the cap is split in two.
    Order inside a year is lost."""
    import json
    import math
    import urllib.parse
    d = out_dir("us_wildfire_data")
    q = "https://apps.fs.usda.gov/ArcX/rest/services/EDW/EDW_FireOccurrenceCurrentEdition_01/MapServer/0/query?"
    stat = json.dumps([{"statisticType": "count", "onStatisticField": "objectid", "outStatisticFieldName": "n"}])

    def sizes(y, lo, hi):
        p = {"where": f"fire_year={y} AND fire_size>={lo} AND fire_size<{hi}", "groupByFieldsForStatistics": "fire_size",
             "outStatistics": stat, "f": "json"}
        res = json.loads(get(q + urllib.parse.urlencode(p)))
        feats = res["features"]
        if res.get("exceededTransferLimit") or len(feats) >= 2000:
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 10
            return sizes(y, lo, mid) + sizes(y, mid, hi)
        return [(y, f["attributes"]["fire_size"], f["attributes"]["n"]) for f in feats]

    rows = []
    for y in range(1992, 2025):
        r = sizes(y, 0, 1e8)
        rows += sorted(r, key=lambda t: t[1])
        print("us_wildfire", y, len(r), "sizes", sum(t[2] for t in r), "fires", flush=True)
    with open(os.path.join(d, "fpa_fod7_year_size_count.csv"), "w") as f:
        f.write("year,fire_size_acres,count\n" + "".join(f"{y},{s!r},{n}\n" for y, s, n in rows))


NWIS_PERENNIAL = {  # flashy rivers that rarely run dry, as possible replacements for Sabino Creek
    "11477000": "Eel River at Scotia CA",
    "11467000": "Russian River near Guerneville CA",
    "08176500": "Guadalupe River at Victoria TX",
    "08167500": "Guadalupe River near Spring Branch TX",
    "02489500": "Pearl River near Bogalusa LA",
}


def nwis_perennial():
    d = out_dir("nwis_river_data")
    for site in NWIS_PERENNIAL:
        url = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd=00060"
               "&startDT=1900-01-01&endDT=2026-10-01")
        open(os.path.join(d, f"{site}_discharge.rdb"), "wb").write(get(url))
        print("nwis", site, NWIS_PERENNIAL[site])


def breach():
    """HHS OCR breach portal: health-data breaches of 500 or more people (2009 to now), both the open cases and the
    archive. The portal is a JSF page, so keep the session cookie, open the HIPAA report page, switch to each tab and
    press the CSV export button."""
    import http.cookiejar
    import re
    import urllib.parse
    d = out_dir("hhs_breach_data")
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    base = "https://ocrportal.hhs.gov/ocr/breach/"

    def post(page, fields, ajax=False):
        h = dict(UA, **({"Faces-Request": "partial/ajax", "X-Requested-With": "XMLHttpRequest"} if ajax else {}))
        req = urllib.request.Request(base + page, data=urllib.parse.urlencode(fields).encode(), headers=h)
        return op.open(req, timeout=300).read()

    state = lambda html: re.search(rb'ViewState:0" value="([^"]+)', html).group(1).decode()
    front = op.open(urllib.request.Request(base + "breach_report.jsf", headers=UA), timeout=120).read()
    page = post("breach_frontpage.jsf", {"ocrForm": "ocrForm", "ocrForm:j_idt39": "ocrForm:j_idt39",
                                         "javax.faces.ViewState": state(front)})
    vs = state(page)
    tabs = re.search(rb'PrimeFaces.cw\("TabView","widget_ocrForm_(\w+)"', page).group(1).decode()
    button = re.search(rb"\{'(ocrForm:\w+)':'ocrForm:\w+'\},''\);return false\"><img id=\"[^\"]+\" src=\"[^\"]+csv", page).group(1).decode()
    tab = f"ocrForm:{tabs}"
    for i, name in ((0, "open"), (1, "archive")):
        if i == 1:                                     # the open cases show first; load the archive tab
            post("breach_report_hip.jsf", {
                f"{tab}_newTab": f"{tab}:archiveTab", "javax.faces.partial.ajax": "true", "javax.faces.source": tab,
                "javax.faces.partial.execute": tab, "javax.faces.partial.render": "ocrForm:breachReports ocrForm:results",
                "javax.faces.behavior.event": "tabChange", "javax.faces.partial.event": "tabChange",
                f"{tab}_contentLoad": "true", f"{tab}_tabindex": "1", f"{tab}_activeIndex": "1",
                "ocrForm": "ocrForm", "javax.faces.ViewState": vs}, ajax=True)
        csv = post("breach_report_hip.jsf", {"ocrForm": "ocrForm", button: button, f"{tab}_activeIndex": str(i),
                                             "javax.faces.ViewState": vs})
        open(os.path.join(d, f"hhs_breach_{name}.csv"), "wb").write(csv)
        print("breach", name, len(csv), "bytes")


def citations(year=2000):
    """OpenAlex: citation counts of all works of type article published in `year`, as (cites, number of articles),
    from the group_by endpoint (paged with a cursor)."""
    import json
    d = out_dir("openalex_citation_data")
    rows, cur = [], "*"
    while cur:
        url = (f"https://api.openalex.org/works?filter=publication_year:{year},type:article&group_by=cited_by_count"
               f"&per-page=200&cursor={cur}")
        r = json.loads(get(url))
        rows += [(int(g["key"]), g["count"]) for g in r["group_by"]]
        cur = r["meta"].get("next_cursor")
        time.sleep(0.2)
    with open(os.path.join(d, f"openalex_articles_{year}_cites.csv"), "w") as f:
        f.write("cites,count\n" + "".join(f"{c},{n}\n" for c, n in sorted(rows)))
    print("citations", year, len(rows), "distinct values", sum(n for _, n in rows), "articles")


def words():
    """Frequencies of the distinct words in Moby Dick (Newman 2005, as distributed by Clauset, Shalizi and Newman 2009)."""
    d = out_dir("word_freq_data")
    open(os.path.join(d, "mobydick_word_counts.txt"), "wb").write(get("https://aaronclauset.github.io/powerlaws/data/words.txt"))


JOBS = {"quake": quake, "danish": danish, "goes": goes, "nwis": nwis, "ghcn": ghcn, "crypto_hourly": crypto_hourly,
        "norwegian": norwegian, "storm": storm, "wiki": wiki, "wildfire": wildfire, "ghcn_flash": ghcn_flash,
        "eaglei": eaglei, "binance_trades": binance_trades, "hackernews": hackernews, "snap": snap, "blackout": blackout,
        "gcmt": gcmt, "us_wildfire": us_wildfire, "nwis_perennial": nwis_perennial,
        "breach": breach, "citations": citations, "words": words}

if __name__ == "__main__":
    for n in (sys.argv[1:] or JOBS):
        JOBS[n]()
