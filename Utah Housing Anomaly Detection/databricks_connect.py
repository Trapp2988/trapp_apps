"""Connect to Databricks with Databricks Connect.

Install (inside your venv):
    python -m pip install databricks-connect==16.1.7 pyarrow==19.0.1 python-dotenv

Use a Python version supported by both the selected Databricks Connect release
and your target compute. For example, Connect 16.1.7 requires NumPy <2, so use
Python 3.12 rather than Python 3.14. Check the Databricks Connect compatibility
matrix before selecting a Connect version or compute runtime.

Credentials: use ONE of these.

A) Environment variables, e.g. in a .env file (keep it out of git):
       DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
       DATABRICKS_TOKEN=<personal access token>
       DATABRICKS_CLUSTER_ID=<cluster id>              # classic cluster
   or, for serverless instead of a cluster:
       DATABRICKS_SERVERLESS_COMPUTE_ID=auto

B) A profile in ~/.databrickscfg, selected with profile="NAME" or the
   DATABRICKS_CONFIG_PROFILE env var. The profile can hold host, auth and
   cluster_id (or serverless_compute_id = auto).

Quick test:
    python databricks_connect.py
    python databricks_connect.py --serverless
    python databricks_connect.py --profile DEFAULT
"""
from __future__ import annotations

import argparse
import os

from databricks.connect import DatabricksSession
from databricks.sdk.core import Config
from logging_utils import configure_logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def get_spark(
    profile: str | None = None,
    serverless: bool | None = None,
    cluster_id: str | None = None,
    host: str | None = None,
    token: str | None = None,
):
    """Return a Spark session running on Databricks.

    Any argument left as None falls back to the matching environment variable
    (or the profile). With no arguments and no env vars, the DEFAULT profile
    in ~/.databrickscfg is used.
    """
    if serverless is None:
        serverless = os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID") == "auto"

    if serverless:
        # Serverless and a cluster id conflict, so serverless wins.
        os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = "auto"
        os.environ.pop("DATABRICKS_CLUSTER_ID", None)
        cluster_id = None

    explicit = {
        "profile": profile,
        "host": host,
        "token": token,
        "cluster_id": cluster_id,
    }
    explicit = {k: v for k, v in explicit.items() if v}

    if explicit:
        config = Config(**explicit)
        return DatabricksSession.builder.sdkConfig(config).getOrCreate()
    return DatabricksSession.builder.getOrCreate()


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the Databricks Connect session")
    ap.add_argument("--profile", help="profile name from ~/.databrickscfg")
    ap.add_argument("--serverless", action="store_true", help="use serverless compute")
    ap.add_argument("--cluster-id", help="classic cluster id")
    args = ap.parse_args()

    app_logger, log_path = configure_logging("databricks_connect")
    app_logger.info("Starting Databricks Connect session")
    app_logger.info("Profile: %s", args.profile or os.getenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT"))
    app_logger.info("Compute: %s", "serverless" if args.serverless else (args.cluster_id or "profile/default"))
    app_logger.info("Log file: %s", log_path)
    try:
        spark = get_spark(
            profile=args.profile,
            serverless=True if args.serverless else None,
            cluster_id=args.cluster_id,
        )
        app_logger.info("Connected. Spark version: %s", spark.version)
        row = spark.sql(
            "SELECT current_user() AS user, current_catalog() AS catalog"
        ).first()
        app_logger.info("Databricks identity and catalog: %s", row)
    except Exception:
        app_logger.exception("Databricks Connect session failed")
        raise


if __name__ == "__main__":
    main()