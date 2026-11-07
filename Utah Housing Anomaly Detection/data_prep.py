import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Modeling schema
# ----------------------------------------------------------------------------
TARGET = "PRICE"
NUMERIC_FEATURES = [
    "BEDS", "BATHS", "SQUARE FEET", "LOT SIZE", "YEAR BUILT",
    "DAYS ON MARKET", "HOA/MONTH", "LATITUDE", "LONGITUDE",
]
CATEGORICAL_FEATURES = ["PROPERTY TYPE", "CITY"]
KEEP_COLUMNS = [
    TARGET,
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    "URL",
    "ZIP OR POSTAL CODE",
]
KEEP_PROPERTY_TYPES = [
    "Single Family Residential",
    "Condo/Co-op",
    "Townhouse",
    "Mobile/Manufactured Home",
    "Apartment",
    "Multi-Family",
    "Other",
]
ANOMALY_TABLE_COLUMNS = [
    "URL", "CITY", "YEAR BUILT", "DAYS ON MARKET", "PRICE", "SQUARE FEET",
    "BEDS", "BATHS", "LOT SIZE", "HOA/MONTH", "LATITUDE", "LONGITUDE",
    "pred_price", "residual", "anomaly_type",
]
ZILLOW_COLUMN_ALIASES = {
    "PRICE": ("price", "unformattedPrice", "listPrice", "hdpData.homeInfo.price"),
    "BEDS": ("beds", "bedrooms", "hdpData.homeInfo.bedrooms"),
    "BATHS": ("baths", "bathrooms", "hdpData.homeInfo.bathrooms"),
    "SQUARE FEET": (
        "livingArea", "area", "sqft", "livingAreaValue",
        "hdpData.homeInfo.livingArea",
    ),
    "LOT SIZE": (
        "lotAreaValue", "lotSize", "lotSizeSqFt",
        "hdpData.homeInfo.lotAreaValue",
    ),
    "YEAR BUILT": ("yearBuilt", "hdpData.homeInfo.yearBuilt"),
    "DAYS ON MARKET": ("daysOnZillow", "days_on_zillow"),
    "HOA/MONTH": (
        "hoaFee", "hoaFee.value", "hdpData.homeInfo.hoaFee",
        "hdpData.homeInfo.hoaFee.value",
    ),
    "LATITUDE": ("latitude", "latLong.latitude", "hdpData.homeInfo.latitude"),
    "LONGITUDE": ("longitude", "latLong.longitude", "hdpData.homeInfo.longitude"),
    "PROPERTY TYPE": ("homeType", "propertyType", "hdpData.homeInfo.homeType"),
    "CITY": ("city", "addressCity", "hdpData.homeInfo.city"),
    "URL": ("detailUrl", "url", "hdpData.homeInfo.detailUrl"),
    "ZIP OR POSTAL CODE": (
        "zipcode", "addressZipcode", "hdpData.homeInfo.zipcode",
    ),
}
LOT_AREA_UNIT_ALIASES = ("lotAreaUnit", "hdpData.homeInfo.lotAreaUnit")

ZILLOW_PROPERTY_TYPES = {
    "SINGLE_FAMILY": "Single Family Residential",
    "SINGLE FAMILY": "Single Family Residential",
    "TOWNHOUSE": "Townhouse",
    "TOWNHOME": "Townhouse",
    "CONDO": "Condo/Co-op",
    "CONDO/CO-OP": "Condo/Co-op",
    "APARTMENT": "Apartment",
    "MANUFACTURED": "Mobile/Manufactured Home",
    "MOBILE": "Mobile/Manufactured Home",
    "MOBILE/MANUFACTURED": "Mobile/Manufactured Home",
    "MULTI_FAMILY": "Multi-Family",
    "MULTI FAMILY": "Multi-Family",
    "OTHER": "Other",
}


def _find_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Find a source column without depending on its capitalization."""
    by_lower = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def _normalize_property_type(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    token = text.replace("-", "_").upper()
    return ZILLOW_PROPERTY_TYPES.get(token, text)


def _normalize_lot_size(df: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    lot_size = df["LOT SIZE"]
    unit_column = _find_column(raw.columns, LOT_AREA_UNIT_ALIASES)
    if unit_column is None:
        return lot_size

    units = raw[unit_column].astype("string").str.strip().str.lower()
    acres = units.isin(["acre", "acres", "ac"])
    square_feet = units.isin(
        ["sqft", "sq ft", "square feet", "square foot", "square_feet"]
    )
    normalized = lot_size.where(square_feet)
    normalized.loc[acres] = lot_size.loc[acres] * 43_560
    return normalized


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------
def prep_data(
    raw: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize Zillow search results into the columns used by the model.

    Known Zillow fields are mapped to the existing uppercase modeling schema.
    Optional fields absent from a search response remain null; numeric columns
    with available values are median-imputed so incomplete listings are usable.
    """
    source = raw.rename(columns=column_map).copy() if column_map else raw.copy()
    df = source.copy()

    for canonical, aliases in ZILLOW_COLUMN_ALIASES.items():
        if canonical not in df.columns:
            source_column = _find_column(source.columns, aliases)
            if source_column is not None:
                df[canonical] = source[source_column]

    if "URL" not in df.columns:
        df["URL"] = ""
    for column in KEEP_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if not df["PRICE"].notna().any():
        raise KeyError(
            "Could not find numeric listing prices. Expected a Zillow 'price' "
            "field or a PRICE column."
        )

    df["LOT SIZE"] = _normalize_lot_size(df, source)
    df["PROPERTY TYPE"] = df["PROPERTY TYPE"].map(_normalize_property_type)
    df["CITY"] = df["CITY"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["CITY"] == "", "CITY"] = "Unknown"
    df["ZIP OR POSTAL CODE"] = df["ZIP OR POSTAL CODE"].fillna("").astype(str)

    df["URL"] = df["URL"].fillna("").astype(str)
    relative_url = df["URL"].str.startswith("/")
    df.loc[relative_url, "URL"] = (
        "https://www.zillow.com" + df.loc[relative_url, "URL"]
    )

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df = df[~df["PROPERTY TYPE"].str.upper().isin({"LOT", "LAND", "VACANT LAND"})]

    missing = df[KEEP_COLUMNS].isna().sum()
    for column in NUMERIC_FEATURES:
        median = df[column].median()
        if pd.notna(median):
            df[column] = df[column].fillna(median)

    return df[KEEP_COLUMNS].reset_index(drop=True), missing
