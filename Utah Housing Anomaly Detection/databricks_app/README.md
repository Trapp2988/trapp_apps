# Utah Housing Anomalies Databricks App

This Streamlit app reads analysis runs from
`workspace.utah_housing.housing_anomalies` and displays a filterable map and
listing table. Its run selector, filters, and full-width minimum/maximum price
controls are in the main page for easier mobile use. Price bounds are inclusive.
Map markers use Esri World Street Map tiles, so users need internet access to
load the basemap.

## Deploy in Databricks

1. Create a Databricks App and deploy the contents of this folder.
2. Add a SQL warehouse app resource with the resource key `sql-warehouse` and
   grant the app **Can use** permission.
3. Grant the app service principal `USE CATALOG`, `USE SCHEMA`, and `SELECT` on
   `workspace.utah_housing.housing_anomalies`.
4. Deploy or restart the app. It uses the app's managed OAuth identity; no
   personal access token or API key is needed.

The warehouse ID is injected through `app.yaml`. Set `ANOMALY_TABLE` there if
the table has a different fully-qualified name.
