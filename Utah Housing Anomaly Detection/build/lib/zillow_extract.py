"""Pull Zillow listings via OpenWeb Ninja, cache raw JSON, flag anomalies.

Setup:
    pip install requests pandas numpy scikit-learn
    Set OWN_API_KEY or API_KEY in the environment or this project's .env file.

Usage:
    python zillow_extract.py pull --location 84101 --pages 1
    python zillow_extract.py detect
"""
import argparse
import json
import logging
import os
import pathlib
import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest

from logging_utils import configure_logging

BASE = "https://api.openwebninja.com/realtime-real-estate-data"
RAW = pathlib.Path("raw")
RAW.mkdir(exist_ok=True)
LISTINGS_CSV = "listings.csv"
logger = logging.getLogger("zillow_extract")

# Field names vary by response, so detect() picks the first candidate that exists.
CANDIDATES = {
    "price": ["price", "unformattedPrice", "listPrice"],
    "sqft": ["livingArea", "area", "sqft", "livingAreaValue"],
    "beds": ["bedrooms", "beds"],
    "baths": ["bathrooms", "baths"],
    "dom": ["daysOnZillow", "days_on_zillow"],
    "zip": ["zipcode", "address.zipcode", "addressZipcode"],
    "type": ["homeType", "propertyType"],
}


def get_api_key():
    """Read the OpenWeb Ninja key from the environment or this project's .env."""
    for name in ("OWN_API_KEY", "API_KEY"):
        if os.environ.get(name):
            return os.environ[name]

    env_files = (
        pathlib.Path.cwd() / ".env",
        pathlib.Path(__file__).with_name(".env"),
    )
    for env_file in env_files:
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in {"OWN_API_KEY", "API_KEY"}:
                value = value.strip().strip("\"'")
                if value:
                    return value

    raise RuntimeError(
        "OpenWeb Ninja API key not found; set OWN_API_KEY or API_KEY, "
        "or add one of those variables to .env."
    )


def fetch_page(location, page, status="FOR_SALE", lo=0, hi=None, refresh=False):
    """One API call per page (up to 41 results). Cached so reruns cost nothing."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", location)
    f = RAW / f"search_{safe}_{status}_{lo}_{hi}_p{page}.json"
    if f.exists() and not refresh:
        logger.info("Using cached Zillow response: %s", f)
        return json.loads(f.read_text())
    params = {"location": location, "page": page, "home_status": status}
    if lo:
        params["min_price"] = lo
    if hi is not None:
        params["max_price"] = hi
    logger.info(
        "Requesting Zillow listings for %s, status=%s, page=%s",
        location,
        status,
        page,
    )
    r = requests.get(
        f"{BASE}/zillow/search",
        headers={"x-api-key": get_api_key()},
        params=params,
        timeout=30,
    )
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        if r.status_code == 401:
            raise RuntimeError(
                "OpenWeb Ninja rejected the API credentials (HTTP 401). "
                "Verify OWN_API_KEY/API_KEY and check the key's status and "
                "remaining quota in your OpenWeb Ninja account."
            ) from exc
        if r.status_code == 403:
            raise RuntimeError(
                "OpenWeb Ninja denied this Zillow request (HTTP 403). "
                "Check that the account and API plan allow this endpoint."
            ) from exc
        if r.status_code == 429:
            raise RuntimeError(
                "OpenWeb Ninja rate or usage limit reached (HTTP 429). "
                "Wait for the limit to reset or reduce --max-pages."
            ) from exc
        raise
    f.write_text(r.text)
    logger.info("Cached Zillow response to %s", f)
    time.sleep(0.15)  # stay under 10 requests/second
    return r.json()


def find_listings(obj):
    """Find the list of property dicts (the ones with a zpid) wherever it sits."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj) and any("zpid" in x for x in obj):
            return obj
        for x in obj:
            got = find_listings(x)
            if got:
                return got
    elif isinstance(obj, dict):
        for v in obj.values():
            got = find_listings(v)
            if got:
                return got
    return []


def pull(location, pages, status):
    rows = []
    for p in range(1, pages + 1):
        found = find_listings(fetch_page(location, p, status))
        if not found:
            break
        rows += found
    df = pd.json_normalize(rows)
    df["query_location"] = location
    df["query_status"] = status
    if os.path.exists(LISTINGS_CSV):
        df = pd.concat([pd.read_csv(LISTINGS_CSV), df]).drop_duplicates("zpid")
    df.to_csv(LISTINGS_CSV, index=False)
    logger.info("%s listings in %s", len(df), LISTINGS_CSV)
    logger.info("Columns: %s", list(df.columns))


def pull_range(
    location,
    status,
    lo,
    hi,
    max_pages,
    depth=0,
    refresh=False,
    on_page=None,
):
    """Page through a query and split only when it reaches the page limit.

    HTTP errors propagate immediately; they must not be mistaken for evidence
    of pagination truncation. Price ranges are recombined without a price cap.
    """
    rows, truncated = [], True
    for p in range(1, max_pages + 1):
        found = find_listings(
            fetch_page(location, p, status, lo, hi, refresh=refresh)
        )
        if not found:
            truncated = False
            break
        if on_page is not None:
            on_page(location, status, found)
        rows += found
    logger.info(
        "%s $%s-%s: %s rows, truncated=%s",
        location,
        lo,
        hi,
        len(rows),
        truncated,
    )
    if truncated and depth < 6:
        top = hi if hi is not None else 5_000_000
        mid = (lo + top) // 2
        rows += pull_range(
            location, status, lo, mid, max_pages, depth + 1, refresh, on_page
        )
        rows += pull_range(
            location, status, mid, hi, max_pages, depth + 1, refresh, on_page
        )
    return rows


def pull_counties(
    counties,
    status,
    max_pages,
    *,
    output_csv=LISTINGS_CSV,
    merge_existing=True,
    refresh=False,
    on_page=None,
):
    """Fetch unique county listings and optionally persist each fetched page."""
    all_rows = []
    for county in counties:
        logger.info("Fetching listings for %s", county)
        rows = pull_range(
            county,
            status,
            0,
            None,
            max_pages,
            refresh=refresh,
            on_page=on_page,
        )
        for r in rows:
            r["query_county"] = county
        all_rows += rows
    if not all_rows:
        logger.warning("No Zillow listings returned for counties: %s", counties)
        return pd.DataFrame()
    df = pd.json_normalize(all_rows).drop_duplicates("zpid")
    df["query_status"] = status
    df["pulled_at"] = date.today().isoformat()
    if merge_existing and output_csv and os.path.exists(output_csv):
        old = pd.read_csv(output_csv)
        df = pd.concat([old, df]).drop_duplicates("zpid", keep="last")
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info("%s unique listings in %s", len(df), output_csv)
    else:
        logger.info("%s unique listings fetched", len(df))
    if "query_county" in df.columns:
        logger.info("Listings by county:\n%s", df["query_county"].value_counts().to_string())
    return df


def pick(df, key):
    for c in CANDIDATES[key]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce") if key != "zip" and key != "type" else df[c]
    logger.warning("No column found for '%s'; edit CANDIDATES", key)
    return pd.Series(np.nan, index=df.index)


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    if not mad:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def detect():
    df = pd.read_csv(LISTINGS_CSV)
    f = pd.DataFrame({
        "zpid": df["zpid"],
        "price": pick(df, "price"),
        "sqft": pick(df, "sqft"),
        "beds": pick(df, "beds"),
        "baths": pick(df, "baths"),
        "dom": pick(df, "dom"),
        "zip": pick(df, "zip"),
        "type": pick(df, "type"),
    })

    # Drop obvious junk and land before modeling
    f = f[(f.price > 0) & (f.sqft > 0)]
    f = f[~f.type.astype(str).str.upper().str.contains("LOT|LAND", na=False)].copy()
    f["ppsf"] = f.price / f.sqft

    # Robust z-score of price per sqft within each zip (or overall if zip missing)
    group = f["zip"] if f["zip"].notna().any() else pd.Series(0, index=f.index)
    f["ppsf_z"] = f.groupby(group)["ppsf"].transform(robust_z)

    # Multivariate outliers
    X = f[["price", "sqft", "beds", "baths", "ppsf"]].fillna(f.median(numeric_only=True))
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X)
    f["iso_score"] = -iso.score_samples(X)  # higher = more anomalous

    f["flag"] = (f.ppsf_z.abs() > 3.5) | (f.iso_score > f.iso_score.quantile(0.98))
    out = f.sort_values("iso_score", ascending=False)
    out.to_csv("anomalies.csv", index=False)
    logger.info("Flagged listings:\n%s", out[out.flag].head(20).to_string(index=False))
    logger.info(
        "%s flagged of %s; full results in anomalies.csv",
        int(f.flag.sum()),
        len(f),
    )

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_latest_listings_csv(data_dir: str | pathlib.Path) -> pd.DataFrame:
    """Load listings.csv or the newest Zillow CSV in data_dir."""
    data_path = pathlib.Path(data_dir)
    listings_file = data_path / LISTINGS_CSV
    if listings_file.is_file():
        logger.info("Loaded listings file: %s", listings_file.name)
        return pd.read_csv(listings_file)

    csv_files = list(data_path.glob("zillow_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No {LISTINGS_CSV} or zillow_*.csv files found in {data_dir}"
        )
    latest = max(csv_files, key=lambda f: f.stat().st_mtime)
    logger.info("Loaded latest listings file: %s", latest.name)
    return pd.read_csv(latest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--location", required=True, help="zip, city, county, or neighborhood")
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--status", default="FOR_SALE", choices=["FOR_SALE", "FOR_RENT", "RECENTLY_SOLD"])
    c = sub.add_parser("pull-counties")
    c.add_argument("--counties", nargs="+", required=True,
                   help='e.g. "Salt Lake County, UT" "Utah County, UT"')
    c.add_argument("--max-pages", type=int, default=20,
                   help="pages per query before splitting by price; use 2 for a cheap test")
    c.add_argument("--status", default="FOR_SALE", choices=["FOR_SALE", "FOR_RENT", "RECENTLY_SOLD"])
    sub.add_parser("detect")
    a = ap.parse_args()
    run_logger, log_path = configure_logging("zillow_extract")
    run_logger.info("Starting Zillow extractor command: %s", a.cmd)
    run_logger.info("Log file: %s", log_path)
    try:
        if a.cmd == "pull":
            pull(a.location, a.pages, a.status)
        elif a.cmd == "pull-counties":
            pull_counties(a.counties, a.status, a.max_pages)
        else:
            detect()
    except Exception:
        run_logger.exception("Zillow extractor command failed")
        raise