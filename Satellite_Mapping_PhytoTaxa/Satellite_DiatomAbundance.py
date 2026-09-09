# -*- coding: utf-8 -*-
"""
PACE OCI - Diatom Concentration Mapping Using SVM
author: Punya
"""

# IMPORT NECESSARY PYTHON PACKAGES
import gc
import joblib
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from joblib import Parallel, delayed

# AUTHENTICATION
auth = earthaccess.login()

# SEARCH FOR PACE OCI DATA
tspan = ("2024-03-15", "2024-03-31")
rrs_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_RRS",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)
chl_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_CHL",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

# PREPROCESSING FUNCTION
def time_from_attr(ds):
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(datetime))
    ds = ds.set_coords("date")
    return ds.sel(
        lat=slice(50, 40),
        lon=slice(-71, -55)
    )

# OPEN PACE DATASETS
# Open Rrs files
fileset1 = earthaccess.open(rrs_results)
# Open chlorophyll files
fileset2 = earthaccess.open(chl_results)
# Open Rrs dataset
dataset1 = xr.open_mfdataset(
    fileset1,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)
# Open chlorophyll dataset
dataset2 = xr.open_mfdataset(
    fileset2,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)

# DETERMINE SEASON
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    elif month in [9, 10, 11]:
        return "Fall"

    else:
        return "Other"

# Determine season from the first date.
# The current analysis period (15-31 March) falls entirely
# within Spring.
date = pd.to_datetime(dataset1.date.values[0])
month = date.month
season = get_season(month)
print(f"Analysis period: {tspan[0]} to {tspan[1]}")
print(f"Season: {season}")

# RENAME CHLOROPHYLL VARIABLE
dataset2 = dataset2.rename(
    {"chlor_a": "HPLCHLA"}
)

# RECHUNK DATA
dataset1 = dataset1.chunk({
    "lat": 40,
    "lon": 40,
    "date": 1
})
dataset2 = dataset2.chunk({
    "lat": 40,
    "lon": 40,
    "date": 1
})

# EXTRACT Rrs AND CHLOROPHYLL
# Extract remote sensing reflectance
rrs = dataset1["Rrs"]
# Extract chlorophyll-a
chl = dataset2["HPLCHLA"]
# Keep only positive chlorophyll values before log transformation
chl = chl.where(
    chl > 0
)
# Convert chlorophyll-a to log10 scale
chl = np.log10(chl)
# Remove non-positive Rrs values
rrs = rrs.where(
    rrs > 0
)

# SELECT VISIBLE WAVELENGTH RANGE
rrs_visible = rrs.sel(
    wavelength=slice(400, 701)
)
# TEMPORAL AVERAGING
# The trained SVM model uses temporally averaged Rrs
# and chlorophyll-a as predictors.
rrs_mean_time_series = rrs_visible.mean(
    dim="date"
)
chl_mean_time_series = chl.mean(
    dim="date"
)
# SELECT SPECTRAL BANDS
selected_wavelengths = [563,415,495,525,
                    472,400,588,442,457,427,
                    507,485,542,435,691,450,
                    465,408]

refl_subset = rrs_mean_time_series.sel(
    wavelength=selected_wavelengths
)

# ALIGN Rrs AND CHLOROPHYLL SPATIALLY
refl_subset, chl_mean_time_series = xr.align(
    refl_subset,
    chl_mean_time_series,
    join="inner"
)
# BUILD PREDICTOR DATASET
band_vars = {
    str(int(w)): refl_subset
    .sel(wavelength=w)
    .drop_vars("wavelength")
    for w in refl_subset.wavelength.values
}
# Add temporally averaged log10 chlorophyll-a
band_vars["HPLCHLA"] = chl_mean_time_series
# Combine all predictor variables
combined_ds = xr.Dataset(
    band_vars
).astype(np.float32)

# CONVERT XARRAY DATA TO DASK DATAFRAME
stacked = (
    combined_ds
    .to_array("variable")
    .transpose("lat", "lon", "variable")
)

# Stack latitude and longitude into a single pixel dimension
pixels = stacked.stack(
    pixel=("lat", "lon")
)
# Get predictor variable names
band_labels = list(
    combined_ds.data_vars
)
# Convert to Dask DataFrame
df = dd.from_dask_array(
    pixels.transpose("pixel", "variable").data,
    columns=band_labels
)
# HANDLE MISSING VALUES
# Calculate global mean for each predictor variable
global_means = df.mean().compute()
def clean_partition(df):
    """
    Remove pixels containing more than 50% missing values
    and fill remaining missing values using global means.
    """
    # Calculate fraction of missing values per pixel
    nan_fraction = df.isna().mean(axis=1)

    # Keep pixels with <= 50% missing values
    df = df[nan_fraction <= 0.5]

    # Fill remaining missing values with global means
    return df.fillna(global_means)

df_clean = df.map_partitions(
    clean_partition
)

# ADD SEASONAL FEATURES
df_clean["Season_Spring"] = np.int8(
    season == "Spring"
)
df_clean["Season_Summer"] = np.int8(
    season == "Summer"
)
df_clean["Season_Fall"] = np.int8(
    season == "Fall"
)
df_clean["Season_Other"] = np.int8(
    season == "Other"
)

# DEFINE EXPECTED SEASONAL FEATURES
expected_cols = [
    "Season_Spring",
    "Season_Summer",
    "Season_Fall",
    "Season_Other"
]
# Ensure all expected seasonal columns exist
for col in expected_cols:

    if col not in df_clean.columns:
        df_clean[col] = np.int8(0)
# LOAD TRAINED SVM MODEL AND SCALER
model = joblib.load(
    "Diatom_SVMmodel.joblib"
)
scaler = joblib.load(
    "Diatom_SVMscaler.joblib"
)
# DEFINE MODEL FEATURES
feature_cols = (["563","415","495",
            "525","472","400","588",
            "442","457","427","507",
            "485","542","435","691",
            "450","465","408"]
    + ["HPLCHLA"]
    + expected_cols
)

# Keep only features required by the model
df_clean = df_clean[
    feature_cols
]
# VERIFY FEATURE ORDER
print("\nScaler features:")
print(list(scaler.feature_names_in_))
print("\nCurrent features:")
print(feature_cols)

# Confirm that the current feature order matches
# the order used during model training.
assert list(scaler.feature_names_in_) == feature_cols, (
    "ERROR: Feature names or feature order do not match "
    "the features used during scaler training."
)

# DEFINE SVM PREDICTION FUNCTION
def predict_partition(part):
    """
    Apply the trained scaler and SVM model to one Dask
    partition.
    """
    # Convert Dask partition to pandas DataFrame
    pdf = part.compute()
    if pdf.empty:
        return None
    # Convert predictors to float32
    X = pdf.astype(np.float32)
    # Match the exact feature order used during training
    X = X[
        scaler.feature_names_in_
    ]
    # Scale predictors
    X_scaled = scaler.transform(X)
    # Predict log10 diatom concentration
    pred = model.predict(X_scaled)
    return pred, pdf.index.to_numpy()

# RUN SVM PREDICTION IN PARALLEL
# Convert Dask DataFrame into delayed partitions
partitions = df_clean.to_delayed()

# Run predictions using all available CPU cores
results = Parallel(
    n_jobs=-1,
    backend="loky",
    verbose=10
)(
    delayed(predict_partition)(part)
    for part in partitions
)

# MERGE PREDICTION RESULTS
predictions = []
indices = []
for result in results:
    if result is None:
        continue

    pred, idx = result
    predictions.append(pred)
    indices.append(idx)

# Combine all predicted values
y_pred = np.concatenate(
    predictions
)

# Combine all corresponding pixel indices
valid_index = np.concatenate(
    indices
)

# CREATE SPATIAL PREDICTION ARRAY
n_lat = combined_ds.sizes["lat"]
n_lon = combined_ds.sizes["lon"]
# Initialize spatial prediction array
prediction = np.full(
    n_lat * n_lon,
    np.nan,
    dtype=np.float32
)
# Insert predictions into corresponding pixels
prediction[valid_index] = y_pred
# Reshape to latitude × longitude
prediction = prediction.reshape(
    n_lat,
    n_lon
)

# FREE MEMORY
del df
del df_clean
del stacked
del pixels
gc.collect()

# CONVERT LOG10 PREDICTIONS TO LINEAR CONCENTRATION
prediction_linear = 10 ** prediction

# CREATE DIATOM CONCENTRATION MAP
fig = plt.figure(
    figsize=(6, 5)
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)
# ADD MAP FEATURES
ax.coastlines(
    resolution="50m",
    linewidth=1
)
ax.add_feature(
    cfeature.BORDERS,
    linestyle=":"
)
ax.add_feature(
    cfeature.LAND,
    facecolor="white"
)
ax.add_feature(
    cfeature.OCEAN,
    facecolor="white"
)
# PLOT DIATOM CONCENTRATION
mesh = ax.pcolormesh(
    combined_ds.lon,
    combined_ds.lat,
    prediction_linear,
    cmap="rainbow",
    shading="auto",
    transform=ccrs.PlateCarree(),
    vmin=0.1,
    vmax=10
)
# ADD COLORBAR
cbar = plt.colorbar(
    mesh,
    orientation="vertical",
    pad=0.05,
    aspect=30,
    shrink=0.6
)
cbar.set_label(
    "Diatom Concentration (μg/L)",
    fontsize=12
)
# ADD AXIS LABELS
ax.set_xlabel(
    "Longitude",
    fontsize=12
)
ax.set_ylabel(
    "Latitude",
    fontsize=12
)
# SET MAP EXTENT
ax.set_extent(
    [-71, -55, 40, 50]
)
# DISPLAY MAP
plt.tight_layout()
plt.show()
