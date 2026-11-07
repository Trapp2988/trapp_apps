"""Databricks Streamlit app for exploring housing anomaly runs."""
from __future__ import annotations

import html
import math
import os
import re
from typing import Any

import folium
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

DEFAULT_TABLE = "workspace.utah_housing.housing_anomalies"
ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTRIBUTION = (
    "Tiles © Esri — Sources: Esri, HERE, Garmin, USGS, Intermap, "
    "and the GIS User Community"
)
ANOMALY_COLUMNS = [
    "run_id",
    "url",
    "city",
    "year_built",
    "days_on_market",
    "price",
    "square_feet",
    "beds",
    "baths",
    "lot_size",
    "hoa_month",
    "latitude",
    "longitude",
    "pred_price",
    "residual",
    "anomaly_type",
]


def _table_name() -> str:
    """Return a fully qualified table identifier from trusted app config."""
    table = os.getenv("ANOMALY_TABLE", DEFAULT_TABLE)
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    if not re.fullmatch(rf"{identifier}\.{identifier}\.{identifier}", table):
        raise ValueError(
            "ANOMALY_TABLE must be a fully qualified catalog.schema.table name."
        )
    return table


def _connection():
    """Create an OAuth-authenticated SQL connection using the app identity."""
    host = os.getenv("DATABRICKS_HOST", "").removeprefix("https://").rstrip("/")
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not host or not warehouse_id:
        raise RuntimeError(
            "Databricks app configuration is incomplete. Add a SQL warehouse "
            "resource with key 'sql-warehouse' and ensure DATABRICKS_HOST is set."
        )

    config = Config(
        host=f"https://{host}",
        client_id=os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
    )
    return sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: config.authenticate,
    )


def _query_frame(query: str, parameters: list[Any] | None = None) -> pd.DataFrame:
    """Run a parameterized SQL query and return its rows as a DataFrame."""
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters or [])
            columns = [description[0] for description in cursor.description or []]
            return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)


@st.cache_data(ttl=300, show_spinner=False)
def _available_runs(table: str) -> pd.DataFrame:
    return _query_frame(
        f"""
        SELECT run_id, COUNT(*) AS anomaly_count
        FROM {table}
        GROUP BY run_id
        ORDER BY run_id DESC
        LIMIT 50
        """
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_run(table: str, run_id: str) -> pd.DataFrame:
    columns = ", ".join(ANOMALY_COLUMNS)
    return _query_frame(
        f"SELECT {columns} FROM {table} WHERE run_id = ?",
        [run_id],
    )


def _make_map(anomalies: pd.DataFrame) -> folium.Map | None:
    valid = anomalies.copy()
    valid["latitude"] = pd.to_numeric(valid["latitude"], errors="coerce")
    valid["longitude"] = pd.to_numeric(valid["longitude"], errors="coerce")
    valid = valid.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return None

    center = [valid["latitude"].mean(), valid["longitude"].mean()]
    map_view = folium.Map(
        location=center,
        zoom_start=9,
        tiles=ESRI_TILES,
        attr=ESRI_ATTRIBUTION,
    )
    colors = {"High-price anomaly": "red", "Low-price anomaly": "blue"}
    cluster = MarkerCluster(name="Anomalies").add_to(map_view)

    for row in valid.to_dict(orient="records"):
        raw_type = row.get("anomaly_type")
        anomaly_type = (
            "Unknown"
            if pd.isna(raw_type) or not str(raw_type).strip()
            else str(raw_type).strip()
        )
        price = pd.to_numeric(row.get("price"), errors="coerce")
        sqft = pd.to_numeric(row.get("square_feet"), errors="coerce")
        raw_city = row.get("city")
        city = html.escape(
            "Unknown"
            if pd.isna(raw_city) or not str(raw_city).strip()
            else str(raw_city).strip()
        )
        url = str(row.get("url") or "")
        safe_url = html.escape(url, quote=True)
        details = [
            f"<b>{html.escape(anomaly_type)}</b>",
            f"City: {city}",
            f"Price: ${price:,.0f}" if pd.notna(price) else "Price: N/A",
            f"Area: {sqft:,.0f} sq ft" if pd.notna(sqft) else "Area: N/A",
        ]
        if safe_url.startswith(("https://www.zillow.com/", "https://zillow.com/")):
            details.append(f'<a href="{safe_url}" target="_blank">Open Zillow listing</a>')
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=6,
            color=colors.get(anomaly_type, "gray"),
            fill=True,
            fill_opacity=0.85,
            tooltip=f"{anomaly_type} — {city}",
            popup=folium.Popup("<br>".join(details), max_width=300),
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(map_view)
    return map_view


def _filter_anomalies(
    data: pd.DataFrame,
    anomaly_types: list[str],
    cities: list[str],
    min_price: int | None,
    max_price: int | None,
) -> pd.DataFrame:
    """Apply selected anomaly, city, and inclusive price filters."""
    prices = pd.to_numeric(data["price"], errors="coerce")
    mask = (
        data["anomaly_type"].astype(str).isin(anomaly_types)
        & data["city"].astype(str).isin(cities)
    )
    if min_price is not None:
        mask &= prices >= min_price
    if max_price is not None:
        mask &= prices <= max_price
    return data[mask].copy()


def main() -> None:
    st.set_page_config(
        page_title="Utah Housing Anomalies",
        page_icon="🏠",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1rem 0.75rem 2rem;
            }
            h1 {
                font-size: 1.65rem !important;
                line-height: 1.2 !important;
            }
            div[data-testid="stCustomComponentV1"] iframe {
                height: 65vh !important;
                min-height: 380px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Utah Housing Price Anomalies")
    st.caption(
        "Explore anomaly runs from Unity Catalog. Map tiles require an internet connection."
    )

    try:
        table = _table_name()
        runs = _available_runs(table)
    except Exception as exc:
        st.error(f"Could not load anomaly runs: {exc}")
        st.stop()

    if runs.empty:
        st.info(f"No anomaly runs found in `{table}`.")
        st.stop()

    run_options = runs["run_id"].astype(str).tolist()
    run_summary = runs.set_index(runs["run_id"].astype(str))
    selected_run = st.selectbox("Analysis run", run_options)
    st.caption(
        f"{int(run_summary.loc[selected_run, 'anomaly_count']):,} anomalies in this run"
    )
    if st.button("Refresh data"):
        _available_runs.clear()
        _load_run.clear()
        st.rerun()

    try:
        data = _load_run(table, selected_run)
    except Exception as exc:
        st.error(f"Could not load anomaly data for run {selected_run}: {exc}")
        st.stop()

    if data.empty:
        st.warning("This run has no anomaly records.")
        st.stop()

    anomaly_types = sorted(data["anomaly_type"].dropna().astype(str).unique())
    cities = sorted(data["city"].dropna().astype(str).unique())
    prices = pd.to_numeric(data["price"], errors="coerce")
    valid_prices = prices.dropna()
    min_price: int | None = None
    max_price: int | None = None

    with st.expander("Filters", expanded=True):
        selected_types = st.multiselect(
            "Anomaly type",
            anomaly_types,
            default=anomaly_types,
            key=f"anomaly_types_{selected_run}",
        )
        selected_cities = st.multiselect(
            "City",
            cities,
            default=cities,
            key=f"cities_{selected_run}",
        )
        if not valid_prices.empty:
            default_min = max(0, math.floor(float(valid_prices.min())))
            ceiling = max(default_min, math.ceil(float(valid_prices.max())))
            min_price = st.number_input(
                "Minimum price",
                min_value=0,
                max_value=ceiling,
                value=default_min,
                step=50_000,
                format="$%d",
                key=f"min_price_{selected_run}",
            )
            max_price = st.number_input(
                "Maximum price",
                min_value=0,
                max_value=ceiling,
                value=ceiling,
                step=50_000,
                format="$%d",
                key=f"max_price_{selected_run}",
            )

    if min_price is not None and max_price is not None and min_price > max_price:
        st.error("Minimum price must be less than or equal to maximum price.")
        st.stop()

    filtered = _filter_anomalies(
        data,
        selected_types,
        selected_cities,
        min_price,
        max_price,
    )

    total, high, low = st.columns(3)
    total.metric("Visible anomalies", f"{len(filtered):,}")
    high.metric(
        "High-price",
        f"{filtered['anomaly_type'].eq('High-price anomaly').sum():,}",
    )
    low.metric(
        "Low-price",
        f"{filtered['anomaly_type'].eq('Low-price anomaly').sum():,}",
    )

    map_view = _make_map(filtered)
    if map_view is None:
        st.warning("No filtered anomalies have coordinates to map.")
    else:
        st_folium(
            map_view,
            height=560,
            use_container_width=True,
            returned_objects=[],
        )

    display_columns = [
        "anomaly_type",
        "city",
        "price",
        "pred_price",
        "residual",
        "square_feet",
        "beds",
        "baths",
        "url",
    ]
    st.subheader("Listings")
    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Zillow listing"),
            "price": st.column_config.NumberColumn("Price", format="$%d"),
            "pred_price": st.column_config.NumberColumn(
                "Predicted price", format="$%d"
            ),
            "residual": st.column_config.NumberColumn("Residual", format="$%d"),
        },
    )


if __name__ == "__main__":
    main()
