
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data_prep import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Scaled numeric features plus dummy-encoded categorical features."""
    numeric_columns = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns and df[column].notna().any()
    ]
    if numeric_columns:
        numeric_values = df[numeric_columns].fillna(df[numeric_columns].median())
        scaler = StandardScaler()
        numeric = pd.DataFrame(
            scaler.fit_transform(numeric_values),
            columns=numeric_columns,
            index=df.index,
        )
    else:
        numeric = pd.DataFrame(index=df.index)
    categorical = df[CATEGORICAL_FEATURES].fillna("Unknown")
    dummies = pd.get_dummies(categorical, drop_first=True, dtype=float)
    return pd.concat([numeric, dummies], axis=1)


def detect_anomalies(df: pd.DataFrame, n_std: float = 2.0):
    """Fit the regression and flag listings whose residual is beyond n_std.

    Returns (model, scored_df, (lower_threshold, upper_threshold)).
    """
    x = build_design_matrix(df)
    y = df[TARGET]
    model = LinearRegression().fit(x, y)

    scored = df.copy()
    scored["pred_price"] = model.predict(x)
    scored["residual"] = y - scored["pred_price"]

    mean, std = scored["residual"].mean(), scored["residual"].std()
    upper, lower = mean + n_std * std, mean - n_std * std

    scored["price_anomaly"] = (
        (scored["residual"] > upper) | (scored["residual"] < lower)
    ).astype(int)
    scored["anomaly_type"] = np.select(
        [scored["residual"] > upper, scored["residual"] < lower],
        ["High-price anomaly", "Low-price anomaly"],
        default="Normal",
    )
    return model, scored, (lower, upper)
