"""Databricks serverless Job entry point for the housing anomaly pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from data_prep import ANOMALY_TABLE_COLUMNS
from data_prep import prep_data
from logging_utils import configure_logging
from pipeline import run
from zillow_extract import pull_counties

SOURCE_TABLE = "workspace.utah_housing.zillow_listings"
ANOMALY_TABLE = "workspace.utah_housing.housing_anomalies"
OUTPUT_VOLUME = "/Volumes/workspace/utah_housing/utah_housing"
LOGGER = logging.getLogger("databricks_job")


def _ensure_databricks_cli_on_path() -> None:
    """Expose the WinGet-installed CLI to the SDK's OAuth token provider."""
    if shutil.which("databricks") or shutil.which("databricks.exe"):
        return

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        candidates = sorted(package_root.glob("Databricks.DatabricksCLI_*"))
        for candidate in candidates:
            if (candidate / "databricks.exe").is_file():
                os.environ["PATH"] = (
                    str(candidate) + os.pathsep + os.environ.get("PATH", "")
                )
                return

    raise FileNotFoundError(
        "Databricks CLI was not found on PATH or in its standard WinGet install "
        "location. Install the CLI, sign in with `databricks auth login --profile "
        "UTAH`, and retry."
    )


def _create_spark_session(
    mode: str,
    profile: str | None,
    serverless: bool,
) -> tuple[Any, bool]:
    """Create native Spark in a Job or a remote Connect session locally."""
    if mode == "job":
        return SparkSession.builder.getOrCreate(), False
    if mode == "auto" and os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return SparkSession.builder.getOrCreate(), False

    from databricks.connect import DatabricksSession
    from databricks.sdk.core import Config

    if serverless:
        os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = "auto"
        os.environ.pop("DATABRICKS_CLUSTER_ID", None)
    _ensure_databricks_cli_on_path()
    config = Config(profile=profile) if profile else Config()
    session = DatabricksSession.builder.sdkConfig(config).getOrCreate()
    return session, True


def _upload_artifacts(
    local_root: Path,
    volume: str,
    profile: str | None,
) -> None:
    """Upload locally generated Connect outputs to a Unity Catalog volume."""
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    remote_root = volume.rstrip("/")
    directories = sorted(
        (path for path in local_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    for directory in directories:
        relative = directory.relative_to(local_root).as_posix()
        workspace.files.create_directory(f"{remote_root}/{relative}")

    for local_path in local_root.rglob("*"):
        if not local_path.is_file():
            continue
        relative = local_path.relative_to(local_root).as_posix()
        with local_path.open("rb") as contents:
            workspace.files.upload(
                f"{remote_root}/{relative}",
                contents,
                overwrite=True,
            )


def _table_column_name(name: str) -> str:
    """Convert model column names to Delta-safe snake_case identifiers."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _spark_compatible_listings(raw: pd.DataFrame) -> pd.DataFrame:
    """Make Zillow fields safe for Spark and case-insensitive Delta schemas."""
    compatible = raw.copy()
    for column in compatible.select_dtypes(include=["object"]).columns:
        compatible[column] = compatible[column].map(
            lambda value: (
                None
                if value is None or value is pd.NA
                else json.dumps(value, default=str)
                if isinstance(value, (dict, list))
                else str(value)
            )
        )

    renamed = []
    seen = set()
    for column in compatible.columns:
        candidate = str(column)
        suffix = 2
        while candidate.casefold() in seen:
            candidate = f"{column}__{suffix}"
            suffix += 1
        if candidate != column:
            LOGGER.warning(
                "Renaming case-insensitive duplicate Zillow column %s to %s",
                column,
                candidate,
            )
        seen.add(candidate.casefold())
        renamed.append(candidate)
    compatible.columns = renamed
    return compatible


def _persist_ingest_batch(
    spark: SparkSession,
    table: str,
    run_id: str,
    county: str,
    status: str,
    records: list[dict[str, Any]],
) -> None:
    """Append one fetched page to a durable, JSON-backed Delta recovery table."""
    schema = StructType(
        [
            StructField("run_id", StringType(), nullable=False),
            StructField("property_id", StringType(), nullable=True),
            StructField("query_county", StringType(), nullable=False),
            StructField("query_status", StringType(), nullable=False),
            StructField("fetched_at", StringType(), nullable=False),
            StructField("payload", StringType(), nullable=False),
        ]
    )
    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows = [
        (
            run_id,
            str(record["zpid"]) if record.get("zpid") is not None else None,
            county,
            status,
            fetched_at,
            json.dumps(record, default=str),
        )
        for record in records
    ]
    (
        spark.createDataFrame(rows, schema=schema)
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(table)
    )
    LOGGER.info(
        "Persisted %s fetched listings to %s for run %s",
        f"{len(rows):,}",
        table,
        run_id,
    )


def _load_ingest_run(
    spark: SparkSession,
    table: str,
    ingest_run_id: str,
) -> pd.DataFrame:
    """Rebuild normalized listings from a selected recovery-table run."""
    from pyspark.sql import functions as F

    batch = (
        spark.table(table)
        .where(F.col("run_id") == ingest_run_id)
        .select("payload", "query_county", "query_status", "fetched_at")
        .toPandas()
    )
    if batch.empty:
        raise ValueError(
            f"No saved Zillow listings found in {table} for ingest run "
            f"{ingest_run_id!r}."
        )

    records = []
    for row_number, row in enumerate(batch.itertuples(index=False), start=1):
        try:
            record = json.loads(row.payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Invalid JSON payload in {table}, run {ingest_run_id!r}, "
                f"row {row_number}."
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object in {table}, run {ingest_run_id!r}, "
                f"row {row_number}."
            )
        record["query_county"] = row.query_county
        record["query_status"] = row.query_status
        record["pulled_at"] = row.fetched_at
        records.append(record)

    raw = pd.json_normalize(records)
    if "zpid" in raw.columns:
        raw = raw.drop_duplicates("zpid", keep="last")
    LOGGER.info(
        "Loaded %s unique listings from recovery table %s, run %s",
        f"{len(raw):,}",
        table,
        ingest_run_id,
    )
    return raw


def _anomaly_spark_frame(
    spark: SparkSession,
    anomalies: pd.DataFrame,
    run_id: str,
):
    """Create a Spark DataFrame with an explicit schema, including empty runs."""
    string_columns = {"URL", "CITY", "anomaly_type"}
    fields = [StructField("run_id", StringType(), nullable=False)]
    fields.extend(
        StructField(
            _table_column_name(column),
            StringType() if column in string_columns else DoubleType(),
            nullable=True,
        )
        for column in ANOMALY_TABLE_COLUMNS
    )
    schema = StructType(fields)
    rows = []
    for row in anomalies[ANOMALY_TABLE_COLUMNS].itertuples(index=False, name=None):
        values = [run_id]
        for column, value in zip(ANOMALY_TABLE_COLUMNS, row):
            if pd.isna(value):
                values.append(None)
            elif column in string_columns:
                values.append(str(value))
            else:
                values.append(float(value))
        rows.append(tuple(values))
    return spark.createDataFrame(rows, schema=schema)


def main(argv: list[str] | None = None) -> None:
    """Run as a native Databricks Job or through a local Connect client."""
    parser = argparse.ArgumentParser(
        description="Run housing anomaly detection on Databricks serverless"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "connect", "job"),
        default="auto",
        help="auto-detect a Databricks Job, or explicitly select Connect or Job",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("DATABRICKS_CONFIG_PROFILE", "UTAH"),
        help="Databricks CLI profile for local Connect runs (default: UTAH)",
    )
    parser.add_argument(
        "--serverless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use serverless compute for Connect (default: enabled)",
    )
    parser.add_argument(
        "--source-table",
        default=SOURCE_TABLE,
        help="Unity Catalog Delta table to refresh with current Zillow listings",
    )
    parser.add_argument(
        "--ingest-table",
        help=(
            "Unity Catalog Delta table for per-page recovery data "
            "(default: <source-table>_ingest)"
        ),
    )
    parser.add_argument(
        "--ingest-run-id",
        help=(
            "process a saved run from --ingest-table without making Zillow API "
            "requests"
        ),
    )
    parser.add_argument(
        "--anomaly-table",
        default=ANOMALY_TABLE,
        help="Unity Catalog destination anomaly table",
    )
    parser.add_argument(
        "--output-volume",
        default=OUTPUT_VOLUME,
        help="Unity Catalog volume directory for logs and figures",
    )
    parser.add_argument(
        "--counties",
        nargs="+",
        default=["Salt Lake County, UT", "Utah County, UT"],
        help="counties to fetch from OpenWeb Ninja",
    )
    parser.add_argument(
        "--status",
        choices=("FOR_SALE", "FOR_RENT", "RECENTLY_SOLD"),
        default="FOR_SALE",
        help="Zillow listing status to fetch",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="pages per county before recursively splitting capped results",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="reuse cached Zillow API responses instead of refreshing them",
    )
    args = parser.parse_args(argv)
    if args.max_pages < 1:
        parser.error("--max-pages must be at least 1")

    is_job = args.mode == "job" or (
        args.mode == "auto" and bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))
    )
    volume = Path(args.output_volume) if is_job else Path("output") / "databricks_connect"
    volume.mkdir(parents=True, exist_ok=True)
    logger, log_path = configure_logging("databricks_job", output_dir=volume)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = volume / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ingest_table = args.ingest_table or f"{args.source_table}_ingest"

    logger.info("Starting Utah housing anomaly Databricks Job")
    logger.info("Listings Delta table: %s", args.source_table)
    logger.info("Per-page recovery table: %s", ingest_table)
    logger.info("Anomaly table: %s", args.anomaly_table)
    logger.info("Output volume: %s", args.output_volume)
    logger.info("Log file: %s", log_path)

    try:
        spark, is_connect = _create_spark_session(
            args.mode,
            args.profile,
            args.serverless,
        )
        logger.info(
            "Connected to Spark version %s using %s",
            spark.version,
            "Databricks Connect" if is_connect else "Databricks Job runtime",
        )
        if args.ingest_run_id:
            logger.info(
                "Skipping Zillow API; loading recovery run %s from %s",
                args.ingest_run_id,
                ingest_table,
            )
            raw = _load_ingest_run(spark, ingest_table, args.ingest_run_id)
        else:
            extractor_logger, extractor_log_path = configure_logging(
                "zillow_extract",
                output_dir=volume,
            )
            logger.info("Zillow API log file: %s", extractor_log_path)
            extractor_logger.info(
                "Fetching %s Zillow counties (%s); cache=%s",
                ", ".join(args.counties),
                args.status,
                "enabled" if args.use_cache else "disabled",
            )
            raw = pull_counties(
                args.counties,
                args.status,
                args.max_pages,
                output_csv=None,
                merge_existing=False,
                refresh=not args.use_cache,
                on_page=lambda county, status, records: _persist_ingest_batch(
                    spark,
                    ingest_table,
                    run_id,
                    county,
                    status,
                    records,
                ),
            )
        if raw.empty:
            raise ValueError(
                "The Zillow API returned no listings; the Delta source table "
                "was not changed."
            )
        prepared, _ = prep_data(raw)
        if prepared.empty:
            raise ValueError(
                "The Zillow API returned no listings with valid positive prices; "
                "the Delta source table was not changed."
            )
        logger.info(
            "Fetched %s unique Zillow listings; refreshing %s",
            f"{len(raw):,}",
            args.source_table,
        )
        (
            spark.createDataFrame(_spark_compatible_listings(raw))
            .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(args.source_table)
        )
        logger.info("Refreshed listings Delta table %s", args.source_table)
        source = spark.table(args.source_table)
        raw = source.toPandas()
        logger.info("Loaded %s listings from %s", f"{len(raw):,}", args.source_table)
        if raw.empty:
            raise ValueError(
                f"Listings Delta table {args.source_table} is empty after refresh"
            )

        results = run(
            raw,
            output_dir=run_dir,
            log_output_dir=volume,
        )
        anomaly_frame = _anomaly_spark_frame(
            spark,
            results["anomalies"],
            run_id,
        )
        (
            anomaly_frame.write
            .format("delta")
            .mode("append")
            .saveAsTable(args.anomaly_table)
        )

        logger.info(
            "Appended %s anomalies to %s for run %s",
            f"{len(results['anomalies']):,}",
            args.anomaly_table,
            run_id,
        )
        logger.info("Saved figures and anomaly CSV to %s", run_dir)
        if is_connect:
            logger.info("Uploading logs and run outputs to %s", args.output_volume)
            _upload_artifacts(volume, args.output_volume, args.profile)
    except Exception:
        logger.exception("Databricks Job failed")
        raise


if __name__ == "__main__":
    main()
