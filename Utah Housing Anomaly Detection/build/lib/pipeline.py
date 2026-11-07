"""Utah housing price anomaly detection (multiple linear regression on list price).

Converted from the notebook "Utah Listed Houses Anomaly Detection - Price Based
Multiple Linear Regression Model".

Zillow search results are normalized to the model's uppercase columns by
`data_prep.py`. Other source column names can be supplied with `column_map`.

Library use:
    from pipeline import run
    results = run(df, output_dir="out", column_map={"livingArea": "SQUARE FEET", ...})

Command line:
    utah-housing-anomaly --csv listings.csv --local
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_figures import build_map, plot_predicted_vs_actual
from data_prep import ANOMALY_TABLE_COLUMNS, prep_data
from logging_utils import configure_logging
from model import detect_anomalies

logger = logging.getLogger("pipeline")

# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def run(
    raw: pd.DataFrame,
    output_dir: str | Path = ".",
    n_std: float = 2.0,
    column_map: dict[str, str] | None = None,
    show: bool = False,
    verbose: bool = True,
    log_output_dir: str | Path = "output",
) -> dict:
    """Prep, fit, flag anomalies, and write outputs to output_dir.

    Writes anomalies.csv, predicted_vs_actual.png, anomalies_scatter.png and
    utah_anomalies_map.html. Returns a dict with the model, scored data,
    anomalies table, thresholds, and missing-value counts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configure_logging("pipeline", output_dir=log_output_dir)
    np.random.seed(42)

    data, missing = prep_data(raw, column_map=column_map)
    if verbose:
        logger.info("Missing values per feature:\n%s", missing.to_string())
        logger.info("Dataset has %s listings", f"{len(data):,}")

    model, scored, (lower, upper) = detect_anomalies(data, n_std=n_std)
    anomalies = scored.loc[scored["price_anomaly"] == 1, ANOMALY_TABLE_COLUMNS].copy()
    if verbose:
        logger.info(
            "Detected %s anomalies out of %s listings.",
            len(anomalies),
            len(scored),
        )
        logger.info("Anomaly types:\n%s", anomalies["anomaly_type"].value_counts().to_string())

    anomalies.to_csv(out / "anomalies.csv", index=False)
    logger.info("Saved anomalies table to %s", out / "anomalies.csv")

    fig1 = plot_predicted_vs_actual(scored)
    fig1.savefig(out / "predicted_vs_actual.png", dpi=150)
    fig2 = plot_predicted_vs_actual(scored, highlight_anomalies=True)
    fig2.savefig(out / "anomalies_scatter.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)

    fmap = build_map(anomalies)
    if fmap is not None:
        fmap.save(str(out / "utah_anomalies_map.html"))
        logger.info("Saved anomaly map to %s", out / "utah_anomalies_map.html")

    if verbose:
        logger.info("Saved outputs to %s", out.resolve())
    return {
        "model": model,
        "scored": scored,
        "anomalies": anomalies,
        "thresholds": (lower, upper),
        "missing": missing,
        "map": fmap,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Utah housing price anomaly detection")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="path to one listings CSV")
    src.add_argument("--csv-dir", help="folder containing listings.csv or Zillow CSV files")
    output = ap.add_mutually_exclusive_group()
    output.add_argument(
        "--local",
        action="store_true",
        help="save outputs to ./output relative to the current directory",
    )
    output.add_argument("--out", help="custom output folder (default: current directory)")
    ap.add_argument("--n-std", type=float, default=2.0,
                    help="residual std devs beyond which a listing is flagged")
    args = ap.parse_args(argv)

    output_dir = Path("output") if args.local else (args.out or ".")
    run_logger, log_path = configure_logging("pipeline")
    run_logger.info("Starting Utah housing anomaly pipeline")
    run_logger.info("Log file: %s", log_path)
    run_logger.info("Output directory: %s", Path(output_dir).resolve())
    try:
        input_path = Path(args.csv) if args.csv else None
        if input_path is None:
            raw = load_latest_listings_csv(args.csv_dir)
        else:
            run_logger.info("Loading listings from %s", input_path.resolve())
            raw = pd.read_csv(input_path)
        run(raw, output_dir=output_dir, n_std=args.n_std)
    except Exception:
        run_logger.exception("Pipeline failed")
        raise


def load_latest_listings_csv(data_dir: str | Path) -> pd.DataFrame:
    """Load listings.csv or the newest Zillow CSV from data_dir."""
    data_path = Path(data_dir)
    listings_file = data_path / "listings.csv"
    if listings_file.is_file():
        logger.info("Loaded listings file: %s", listings_file.name)
        return pd.read_csv(listings_file)

    csv_files = list(data_path.glob("zillow_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No listings.csv or zillow_*.csv files found in {data_dir}"
        )
    latest = max(csv_files, key=lambda path: path.stat().st_mtime)
    logger.info("Loaded latest listings file: %s", latest.name)
    return pd.read_csv(latest)


if __name__ == "__main__":
    main()