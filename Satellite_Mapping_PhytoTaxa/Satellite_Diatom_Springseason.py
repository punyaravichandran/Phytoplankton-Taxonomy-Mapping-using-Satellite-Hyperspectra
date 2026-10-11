# -*- coding: utf-8 -*-
"""
PACE OCI - Diatom Biomass Mapping Using SVM
author: Punya
"""

# IMPORT REQUIRED PACKAGES
import gc
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask.dataframe as dd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

# EARTHDATA AUTHENTICATION AND SEARCH PARAMETERS
auth = earthaccess.login()
# Study area: Northwest Atlantic Ocean
bbox = (-71, 40, -55, 50)
# Temporal coverage
tspan = ("2024-03-15", "2024-03-31")

# SEARCH FOR PACE OCI DATA
# PACE spectral reflectance data
rrs_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_SFREFL",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

# PACE biogeochemical data, including chlorophyll-a
chl_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_BGC",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

if not rrs_results:
    raise ValueError("No PACE reflectance granules were found.")

if not chl_results:
    raise ValueError("No PACE chlorophyll granules were found.")

# PREPROCESSING FUNCTION
def time_from_attr(ds):
    """
    Extract the acquisition date from the dataset attributes
    and subset the data to the study area.
    """

    datetime = ds.attrs["time_coverage_start"].replace("Z", "")

    ds["date"] = ((), np.datetime64(datetime))
    ds = ds.set_coords("date")

    lat = ds["lat"]

    # Handle ascending or descending latitude coordinates
    lat_slice = (
        slice(50, 40)
        if lat[0] > lat[-1]
        else slice(40, 50)
    )

    return ds.sel(
        lat=lat_slice,
        lon=slice(-71, -55)
    )

# OPEN PACE DATASETS
fileset1 = earthaccess.open(rrs_results)
fileset2 = earthaccess.open(chl_results)

dataset1 = xr.open_mfdataset(
    fileset1,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)

dataset2 = xr.open_mfdataset(
    fileset2,
    preprocess=time_from_attr,
    combine="nested",
    concat_dim="date",
    parallel=True
)

# DETERMINE SEASON
def get_season(month):
    """Assign a season based on the calendar month."""

    if month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    elif month in [9, 10, 11]:
        return "Fall"

    else:
        return "Other"


date = pd.to_datetime(dataset1.date.values[0])
month = date.month
season = get_season(month)

print("First acquisition date:", date)
print("Assigned season:", season)

# RENAME CHLOROPHYLL VARIABLE AND RECHUNK DATA
dataset2 = dataset2.rename(
    {"chlor_a": "HPLCHLA"}
)

# Rechunk for spatial and temporal processing
dataset1 = dataset1.chunk(
    {"lat": 40, "lon": 40, "date": 1}
)

dataset2 = dataset2.chunk(
    {"lat": 40, "lon": 40, "date": 1}
)


# EXTRACT REFLECTANCE AND CHLOROPHYLL VARIABLES
rrs = dataset1["rhos"]

chl_raw = dataset2["HPLCHLA"]

# Convert chlorophyll-a to log10 space
chl = np.log10(
    chl_raw.where(chl_raw > 0)
)

# Remove non-positive reflectance values
rrs = rrs.where(rrs > 0)

# CALCULATE TEMPORAL MEANS
# Select visible wavelengths between 400 and 701 nm
rrs_visible = rrs.sel(
    wavelength=slice(400, 701)
)

# Average reflectance and chlorophyll over time
rrs_mean_time_series = rrs_visible.mean(
    dim="date"
)

chl_mean_time_series = chl.mean(
    dim="date"
)


# OPTIONAL PUBLICATION-QUALITY REFLECTANCE MAP
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import (
     LONGITUDE_FORMATTER,
     LATITUDE_FORMATTER
 )

 rrs_400 = rrs_mean_time_series.sel(wavelength=400)

 fig = plt.figure(figsize=(8, 7), dpi=300)

 ax = plt.axes(
     projection=ccrs.PlateCarree()
 )

 ax.set_extent(
     [-71, -55, 40, 50],
     crs=ccrs.PlateCarree()
 )

 im = rrs_400.plot(
     ax=ax,
     transform=ccrs.PlateCarree(),
     cmap="viridis",
     add_colorbar=False,
     robust=True
 )

 ax.coastlines(
     resolution="10m",
     linewidth=0.8
 )

 ax.add_feature(
     cfeature.LAND,
     facecolor="white",
     edgecolor="black",
     zorder=100
 )

 ax.add_feature(
     cfeature.BORDERS,
     linewidth=0.5,
     linestyle=":"
 )

 ax.add_feature(
     cfeature.LAKES,
     facecolor="none",
     edgecolor="black",
     linewidth=0.3
 )

 ax.add_feature(
     cfeature.RIVERS,
     linewidth=0.2
 )

 gl = ax.gridlines(
     draw_labels=True,
     linewidth=0.01,
     color="gray",
     alpha=0.1,
     linestyle="--"
 )

 gl.top_labels = False
 gl.right_labels = False
 gl.xformatter = LONGITUDE_FORMATTER
 gl.yformatter = LATITUDE_FORMATTER
 gl.xlabel_style = {"size": 11, "weight": "bold"}
 gl.ylabel_style = {"size": 11, "weight": "bold"}
 gl.xlocator = mticker.MultipleLocator(5)
 gl.ylocator = mticker.MultipleLocator(2)

 cbar = plt.colorbar(
     im,
     ax=ax,
     orientation="vertical",
     pad=0.03,
     shrink=0.60
 )

 cbar.set_label(
     r"Rrs (400 nm) [$sr^{-1}$]",
     fontsize=13,
     fontweight="bold"
 )

 cbar.ax.tick_params(labelsize=11)

 for spine in ax.spines.values():
     spine.set_linewidth(1.2)

 plt.tight_layout()

 plt.savefig(
     "PACE_Rrs400_Publication.png",
     dpi=600,
     bbox_inches="tight"
 )
 plt.show()


# CHLOROPHYLL MAP
 rrs_400 = chl_mean_time_series

 fig = plt.figure(figsize=(8, 7), dpi=300)

 ax = plt.axes(
     projection=ccrs.PlateCarree()
 )

 ax.set_extent(
     [-71, -55, 40, 50],
     crs=ccrs.PlateCarree()
 )

 im = rrs_400.plot(
     ax=ax,
     transform=ccrs.PlateCarree(),
     cmap="viridis",
     add_colorbar=False,
     robust=True
 )

 ax.coastlines(
     resolution="10m",
     linewidth=0.8
 )

 ax.add_feature(
     cfeature.LAND,
     facecolor="white",
     edgecolor="black",
     zorder=100
 )

 ax.add_feature(
     cfeature.BORDERS,
     linewidth=0.5,
     linestyle=":"
 )

 ax.add_feature(
     cfeature.LAKES,
     facecolor="none",
     edgecolor="black",
     linewidth=0.3
 )

 ax.add_feature(
     cfeature.RIVERS,
     linewidth=0.2
 )

 gl = ax.gridlines(
     draw_labels=True,
     linewidth=0.01,
     color="gray",
     alpha=0.1,
     linestyle="--"
 )

 gl.top_labels = False
 gl.right_labels = False
 gl.xformatter = LONGITUDE_FORMATTER
 gl.yformatter = LATITUDE_FORMATTER
 gl.xlabel_style = {"size": 11, "weight": "bold"}
 gl.ylabel_style = {"size": 11, "weight": "bold"}
 gl.xlocator = mticker.MultipleLocator(5)
 gl.ylocator = mticker.MultipleLocator(2)

 cbar = plt.colorbar(
     im,
     ax=ax,
     orientation="vertical",
     pad=0.03,
     shrink=0.60
 )

 cbar.set_label(
     "Chlorophyll Log10 scale",
     fontsize=13,
     fontweight="bold"
 )

 cbar.ax.tick_params(labelsize=11)

 for spine in ax.spines.values():
     spine.set_linewidth(1.2)

 plt.tight_layout()

 plt.savefig(
     "PACE_Chlorophyll_Publication.png",
     dpi=600,
     bbox_inches="tight"
 )
 plt.show()

# SELECT SPECTRAL BANDS FOR THE DIATOM MODEL
selected_wavelengths = [
    560, 415, 495, 525, 475, 400,
    586, 450, 440, 430, 505, 460,
    485, 540, 650, 425
]

refl_subset = rrs_mean_time_series.sel(
    wavelength=selected_wavelengths
)

# ALIGN REFLECTANCE AND CHLOROPHYLL SPATIALLY
refl_subset, chl_mean_time_series = xr.align(
    refl_subset,
    chl_mean_time_series,
    join="inner"
)

# BUILD THE PREDICTOR DATASET
band_vars = {
    str(int(w)): (
        refl_subset
        .sel(wavelength=w)
        .drop_vars("wavelength")
    )
    for w in refl_subset.wavelength.values
}

# Include log10 chlorophyll-a as a model feature
band_vars["HPLCHLA"] = chl_mean_time_series

combined_ds = xr.Dataset(
    band_vars
).astype(np.float32)

# CONVERT XARRAY DATA TO A DASK DATAFRAME
stacked = (
    combined_ds
    .to_array("variable")
    .transpose("lat", "lon", "variable")
)

pixels = stacked.stack(
    pixel=("lat", "lon")
)

band_labels = list(
    combined_ds.data_vars
)

df = dd.from_dask_array(
    pixels.transpose("pixel", "variable").data,
    columns=band_labels
)

# REMOVE MISSING VALUES
global_means = df.mean().compute()


def clean_partition(partition):
    """
    Remove pixels with more than 50% missing features.
    Fill remaining missing values with global feature means.
    """

    nan_fraction = partition.isna().mean(axis=1)

    partition = partition[
        nan_fraction <= 0.5
    ]

    return partition.fillna(global_means)


df = df.map_partitions(
    clean_partition
)

# ADD SEASONAL FEATURES
df["Season"] = season

df["Season"] = df["Season"].astype(
    "category"
)

df["Season_Spring"] = np.int8(
    season == "Spring"
)

df["Season_Summer"] = np.int8(
    season == "Summer"
)

df["Season_Fall"] = np.int8(
    season == "Fall"
)

df["Season_Other"] = np.int8(
    season == "Other"
)

expected_cols = [
    "Season_Spring",
    "Season_Summer",
    "Season_Fall",
    "Season_Other"
]

# Ensure all expected seasonal features exist
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.int8(0)

# LOAD TRAINED MODEL AND SCALERS
model = joblib.load(
    "Diatom_SVMmodel.joblib"
)

scaler = joblib.load(
    "Diatom_SVMscaler.joblib"
)

normalization = joblib.load(
    "Satellite_to_Simulation_MinMaxScaler_2024_AllStations.joblib"
)

print("Model feature names:")
print(scaler.feature_names_in_)

print("\nAvailable dataframe columns:")
print(df.columns)

# DEFINE MODEL FEATURES
feature_cols = (
    [
        "560", "415", "495", "525",
        "475", "400", "586", "450",
        "440", "430", "505", "460",
        "485", "540", "650", "425"
    ]
    + ["HPLCHLA"]
    + expected_cols
)

# Select only the required predictors
df_clean = df[feature_cols]

# SATELLITE SPECTRAL BANDS USED BY THE DIATOM SVM
satellite_band_cols = [
    "560", "415", "495", "525",
    "475", "400", "586", "450",
    "440", "430", "505", "460",
    "485", "540", "650", "425"
]


# CHECK NORMALIZATION PARAMETERS
normalization_wavelengths = np.asarray(
    normalization["wavelengths"],
    dtype=float
)

print(
    "Number of normalization bands:",
    len(normalization_wavelengths)
)

print(
    "Normalization wavelength range:",
    normalization_wavelengths.min(),
    "to",
    normalization_wavelengths.max(),
    "nm"
)


# NORMALIZE SATELLITE REFLECTANCE TO SIMULATED REFLECTANCE
def normalize_satellite_to_simulated(X):
    """
    Transform satellite reflectance to the simulated-reflectance
    distribution using the 2024 all-station normalization parameters.

    The transformation is performed independently for each band.

    Expected normalization parameters:
        satellite_lower
        satellite_upper
        simulation_lower
        simulation_upper
    """

    X = X.copy()

    for col in satellite_band_cols:

        wavelength = float(col)

        if not np.any(
            np.isclose(
                normalization_wavelengths,
                wavelength
            )
        ):
            raise KeyError(
                f"{wavelength} nm is not available in the "
                "normalization joblib."
            )

        # Find the corresponding normalization record
        parameter_record = next(
            (
                p
                for p in normalization["parameters"]
                if np.isclose(
                    float(p["wavelength"]),
                    wavelength
                )
            ),
            None
        )

        if parameter_record is None:
            raise KeyError(
                f"No normalization parameters found "
                f"for {wavelength} nm."
            )

        # Satellite reflectance range
        sat_lo = float(
            parameter_record["satellite_lower"]
        )

        sat_hi = float(
            parameter_record["satellite_upper"]
        )

        # Simulated reflectance range
        sim_lo = float(
            parameter_record["simulation_lower"]
        )

        sim_hi = float(
            parameter_record["simulation_upper"]
        )

        # Validate normalization ranges
        if not np.isfinite(
            [sat_lo, sat_hi, sim_lo, sim_hi]
        ).all():
            raise ValueError(
                f"Invalid normalization parameters "
                f"for {wavelength} nm."
            )

        if np.isclose(sat_hi, sat_lo):
            raise ValueError(
                f"Zero satellite normalization range "
                f"for {wavelength} nm."
            )

        if np.isclose(sim_hi, sim_lo):
            raise ValueError(
                f"Zero simulation normalization range "
                f"for {wavelength} nm."
            )

        # Satellite-to-simulation transformation
        X[col] = (
            sim_lo
            + (
                (X[col] - sat_lo)
                * (sim_hi - sim_lo)
                / (sat_hi - sat_lo)
            )
        )

    return X


# PREPARE DASK PARTITIONS
partitions = df_clean.to_delayed()

# DEFINE THE PARTITION-WISE PREDICTION FUNCTION
def predict_partition(part):
    """
    Process one Dask partition, normalize reflectance,
    scale features, and predict diatom concentration.
    """

    # Convert delayed partition to a pandas DataFrame
    pdf = part.compute()

    if pdf.empty:
        return None

    # Preserve feature names
    X = pdf.astype(
        np.float32
    )

    # Match the exact feature order used during model training
    X = X[
        scaler.feature_names_in_
    ].copy()

    # Step 1: Normalize satellite reflectance to simulated data

    X = normalize_satellite_to_simulated(
        X
    )

    # Step 2: Apply the trained SVM/SVR StandardScaler
    X_scaled = scaler.transform(
        X
    )

   
    # Step 3: Predict diatom concentration

    pred = model.predict(
        X_scaled
    )

    return (
        pred,
        pdf.index.to_numpy()
    )


# RUN PARALLEL PREDICTIONS
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

    predictions.append(
        pred
    )

    indices.append(
        idx
    )

if len(predictions) == 0:
    raise ValueError(
        "No valid predictions were generated."
    )

y_pred = np.concatenate(
    predictions
)

valid_index = np.concatenate(
    indices
)

print(
    "Number of predicted pixels:",
    len(y_pred)
)

print(
    "Prediction range:",
    np.nanmin(y_pred),
    "to",
    np.nanmax(y_pred)
)


# RECONSTRUCT THE SPATIAL PREDICTION MAP
n_lat = combined_ds.sizes["lat"]
n_lon = combined_ds.sizes["lon"]

# Initialize all pixels as missing
prediction = np.full(
    n_lat * n_lon,
    np.nan,
    dtype=np.float32
)

# Insert predictions at their corresponding flattened indices
prediction[valid_index] = y_pred

# Restore latitude-longitude dimensions
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

# MASK EXTREME PREDICTIONS
# Mask predictions greater than 2
prediction = np.ma.masked_where(
    prediction > 2,
    prediction
)

# Mask predictions less than or equal to -5
prediction = np.ma.masked_where(
    prediction <= -5,
    prediction
)

# PLOT THE DIATOM CONCENTRATION MAP
# Predictions remain in log10 space
prediction_plot = np.ma.masked_invalid(
    np.asarray(prediction)
)

fig, ax = plt.subplots(
    figsize=(6, 5),
    dpi=150,
    subplot_kw={
        "projection": ccrs.PlateCarree()
    }
)

# Plot spatial predictions
mesh = ax.pcolormesh(
    combined_ds.lon,
    combined_ds.lat,
    prediction_plot,
    cmap="rainbow",
    shading="auto",
    transform=ccrs.PlateCarree(),
    vmin=-1.5,
    vmax=0.8,
    zorder=1
)

# ADD LAND, COASTLINES, AND BORDERS
# Draw land over the raster
ax.add_feature(
    cfeature.LAND,
    facecolor="white",
    edgecolor="none",
    zorder=3
)

# Add coastlines
ax.coastlines(
    resolution="50m",
    linewidth=0.8,
    color="black",
    zorder=4
)

# Add international and administrative borders
ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    linewidth=0.5,
    edgecolor="black",
    zorder=4
)

# SET GEOGRAPHIC EXTENT
ax.set_extent(
    [-71, -55, 40, 50],
    crs=ccrs.PlateCarree()
)

# ADD COLORBAR
# Tick positions are in log10 space.
# Labels show the corresponding values in normal concentration space.
log_ticks = np.arange(
    -4,
    3,
    0.8
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    pad=0.04,
    shrink=0.6,
    ticks=log_ticks
)

concentration_labels = [
    f"{10.0**x:g}"
    for x in log_ticks
]

cbar.set_ticklabels(
    concentration_labels
)

cbar.set_label(
    "Diatom Concentration (μg/L)",
    fontsize=12
)

# LABEL AXES AND FORMAT FIGURE
ax.set_xlabel(
    "Longitude",
    fontsize=12
)

ax.set_ylabel(
    "Latitude",
    fontsize=12
)

plt.tight_layout()

# SAVE AND DISPLAY THE FINAL MAP
plt.savefig(
    "PACE_Diatom_Concentration_Map.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

print(
    "Diatom concentration prediction map generated successfully."
)
