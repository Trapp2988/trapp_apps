
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

from data_prep import TARGET


# ----------------------------------------------------------------------------
# Plots and map
# ----------------------------------------------------------------------------
def plot_predicted_vs_actual(scored: pd.DataFrame, highlight_anomalies: bool = False):
    """Predicted vs actual price. With highlight_anomalies, anomalies are red."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if highlight_anomalies:
        colors = scored["price_anomaly"].map({1: "red", 0: "grey"})
        ax.scatter(scored[TARGET], scored["pred_price"], c=colors,
                   alpha=0.6, s=30, edgecolors="none")
        ax.set_title("Regression Residual Anomalies\n"
                     "(red = unusual listed price given features)")
    else:
        ax.scatter(scored[TARGET], scored["pred_price"], alpha=0.5)
        ax.set_title("Linear Regression Predicted vs Actual Price")

    lims = [scored[TARGET].min(), scored[TARGET].max()]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    fig.tight_layout()
    return fig


def build_map(anomalies: pd.DataFrame) -> folium.Map | None:
    """Clustered map of anomalies, one layer per anomaly type."""
    cols = ["LATITUDE", "LONGITUDE", "URL", "anomaly_type", "PRICE", "SQUARE FEET"]
    plot_df = anomalies[cols].dropna(subset=["LATITUDE", "LONGITUDE"])
    if plot_df.empty:
        return None

    color_map = {
        "High-price anomaly": "red",
        "Low-price anomaly": "blue",
        "Unknown": "gray",
    }
    center = [plot_df["LATITUDE"].mean(), plot_df["LONGITUDE"].mean()]
    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Sources: Esri, HERE, Garmin, USGS, Intermap, "
            "and the GIS User Community"
        ),
    )

    layers = {}
    for atype in color_map:
        group = folium.FeatureGroup(name=atype, show=True)
        cluster = MarkerCluster(name=f"{atype} cluster")
        cluster.add_to(group)
        group.add_to(m)
        layers[atype] = cluster

    for _, row in plot_df.iterrows():
        raw = row["anomaly_type"]
        atype = "Unknown" if pd.isna(raw) or str(raw).strip() == "" else str(raw).strip()
        if atype not in layers:
            atype = "Unknown"

        price, sqft = row["PRICE"], row["SQUARE FEET"]
        ok = np.isfinite(price) and np.isfinite(sqft)
        tooltip = folium.Tooltip(
            f"{atype} \u2022 ${price:,.0f} \u2022 {int(sqft):,} sqft" if ok else atype
        )
        popup = folium.Popup(
            f"""
            <div style="font-size:14px;">
              <b>{atype}</b><br>
              Price: {f'${price:,.0f}' if np.isfinite(price) else 'N/A'}<br>
              Sq Ft: {f'{sqft:,.0f}' if np.isfinite(sqft) else 'N/A'}<br>
              <a href="{row['URL']}" target="_blank">Open listing</a>
            </div>
            """,
            max_width=300,
        )
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=6,
            color=color_map[atype],
            fill=True,
            fill_opacity=0.85,
            popup=popup,
            tooltip=tooltip,
        ).add_to(layers[atype])

    folium.LayerControl(collapsed=False).add_to(m)
    return m
