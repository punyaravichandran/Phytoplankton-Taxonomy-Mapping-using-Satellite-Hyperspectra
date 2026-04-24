# -*- coding: utf-8 -*-
"""
Satellite Mapping of Diatom Abundance
author: Punya
"""

# IMPORT NECESSARY PYTHON PACKAGES
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# AUTHENTICATION
auth = earthaccess.login()

# SEARCH DATA
bbox = (-71, 40, -55, 50)
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

# PREPROCESS FUNCTION
def preprocess(ds):
    time = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(time, "ns"))
    ds = ds.set_coords("date")
    return ds.sel(lat=slice(50, 40), lon=slice(-71, -55))

# LOAD DATA
rrs_ds = xr.open_mfdataset(
    earthaccess.open(rrs_results),
    preprocess=preprocess,
    concat_dim="date",
    combine="nested",
    parallel=True
)

chl_ds = xr.open_mfdataset(
    earthaccess.open(chl_results),
    preprocess=preprocess,
    concat_dim="date",
    combine="nested",
    parallel=True
).rename({"chlor_a": "HPLCHLA"})

# Chunking
rrs_ds = rrs_ds.chunk({'lat': 20, 'lon': 20, 'date': 1})
chl_ds = chl_ds.chunk({'lat': 20, 'lon': 20, 'date': 1})

# DATA PREPARATION
rrs = rrs_ds["Rrs"].where(rrs_ds["Rrs"] > 0)
chl = np.log10(chl_ds["HPLCHLA"])

# Average over time
rrs_mean = rrs.sel(wavelength=slice(400, 700)).mean(dim="date")
chl_mean = chl.mean(dim="date")

# SELECT BANDS
selected_wavelengths = [
    573, 575, 578, 580, 583, 586, 588,
    613, 615, 618, 620, 623, 625, 627,
    630, 632, 635, 637, 640
]
rrs_subset = rrs_mean.sel(wavelength=selected_wavelengths)

# Align
rrs_subset, chl_mean = xr.align(rrs_subset, chl_mean)
# Convert to dataset
data_vars = {
    f"Refl{int(w)}nm": rrs_subset.sel(wavelength=w).drop_vars("wavelength")
    for w in selected_wavelengths
}
data_vars["HPLCHLA"] = chl_mean
ds = xr.Dataset(data_vars)

# FLATTEN TO ML INPUT
stacked = ds.to_array("feature").stack(pixel=("lat", "lon")).transpose("pixel", "feature")
df = pd.DataFrame(stacked.values, columns=stacked.feature.values)

# Save valid pixel index
valid_idx = df.index

# CLEAN DATA
def clean_data(df, threshold=0.5):
    nan_frac = df.isna().mean(axis=1)
    df = df[nan_frac <= threshold]
    return df.fillna(df.mean())

df_clean = clean_data(df)
if df_clean.empty:
    raise ValueError("No valid pixels after cleaning.")

# LOAD TRAINED MODEL
import joblib
model = joblib.load("Diatom_SVMmodelfinal.joblib")
scaler = joblib.load("DiatomSVMscalerfinal.joblib")

# Ensure feature order
df_clean = df_clean[scaler.feature_names_in_]
# Scale using TRAINED scaler (important)
X_scaled = scaler.transform(df_clean)

# PREDICTION
y_pred = model.predict(X_scaled)

# RECONSTRUCT MAP
n_lat = len(ds.lat)
n_lon = len(ds.lon)
biomass_flat = np.full(len(df), np.nan)
biomass_flat[df_clean.index] = y_pred
biomass_map = biomass_flat.reshape(n_lat, n_lon)

# Mask unrealistic values
biomass_map = np.ma.masked_where((biomass_map > 2) | (biomass_map < -5), biomass_map)

# VISUALIZATION
plt.figure(figsize=(6, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='white')
ax.add_feature(cfeature.BORDERS, linestyle=':')
mesh = ax.pcolormesh(
    ds.lon, ds.lat, biomass_map,
    cmap='viridis',
    shading='auto',
    transform=ccrs.PlateCarree(),
    vmin=-3, vmax=2
)
plt.colorbar(mesh, label="Log Diatom Biomass")
ax.set_extent([-71, -55, 40, 50])
plt.title("PACE OCI Diatom Biomass Prediction")
plt.show()