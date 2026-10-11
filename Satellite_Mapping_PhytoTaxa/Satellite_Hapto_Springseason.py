
# PACE OCI HAPTOPHYTE CONCENTRATION MAPPING
# Satellite Reflectance + Chlorophyll-a + Pre-trained SVM/SVR

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

# EARTHDATA AUTHENTICATION
auth = earthaccess.login()

# DEFINE STUDY AREA AND TIME PERIOD
# Northwest Atlantic Ocean
bbox = (-71, 40, -55, 50)

# March 2024
tspan = ("2024-03-15", "2024-03-31")

# SEARCH PACE OCI DATA
# PACE spectral reflectance
rrs_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_SFREFL",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

# PACE biogeochemical data
chl_results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_BGC",
    temporal=tspan,
    granule_name="*DAY.*.4km.*"
)

if not rrs_results:
    raise ValueError(
        "No PACE reflectance granules were found."
    )

if not chl_results:
    raise ValueError(
        "No PACE biogeochemical granules were found."
    )

print("Reflectance granules:", len(rrs_results))
print("Chlorophyll granules:", len(chl_results))

# PREPROCESSING FUNCTION
def time_from_attr(ds):
    """
    Extract the acquisition date from the dataset attributes
    and subset the data to the study region.
    """

    datetime = ds.attrs["time_coverage_start"].replace("Z", "")

    ds["date"] = ((), np.datetime64(datetime))

    ds = ds.set_coords("date")

    lat = ds["lat"]

    # Account for ascending or descending latitude coordinates
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

print("Reflectance dataset:")
print(dataset1)

print("\nBiogeochemical dataset:")
print(dataset2)

# DETERMINE SEASON
def get_season(month):
    """Return the season corresponding to a calendar month."""

    if month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    elif month in [9, 10, 11]:
        return "Fall"

    else:
        return "Other"


date = pd.to_datetime(
    dataset1.date.values[0]
)

month = date.month
season = get_season(month)

print("First acquisition date:", date)
print("Season:", season)


# RENAME CHLOROPHYLL VARIABLE
dataset2 = dataset2.rename(
    {"chlor_a": "HPLCHLA"}
)

# RECHUNK DATASETS
dataset1 = dataset1.chunk(
    {
        "lat": 40,
        "lon": 40,
        "date": 1
    }
)

dataset2 = dataset2.chunk(
    {
        "lat": 40,
        "lon": 40,
        "date": 1
    }
)


# EXTRACT REFLECTANCE AND CHLOROPHYLL
rrs = dataset1["rhos"]

chl_raw = dataset2["HPLCHLA"]

# Convert positive chlorophyll-a to log10 space
chl = np.log10(
    chl_raw.where(chl_raw > 0)
)

# Retain positive reflectance
rrs = rrs.where(
    rrs > 0
)

# CALCULATE TEMPORAL MEANS
# Visible wavelengths from 400 to 701 nm
rrs_visible = rrs.sel(
    wavelength=slice(400, 701)
)

# Temporal mean reflectance
rrs_mean_time_series = rrs_visible.mean(
    dim="date"
)

# Temporal mean log10 chlorophyll-a
chl_mean_time_series = chl.mean(
    dim="date"
)


# SELECT MODEL SPECTRAL WAVELENGTHS
selected_wavelengths = [
    510, 410, 575, 480,
    445, 540, 460, 400,
    495, 615, 470, 425,
    525, 435, 555, 490
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

# Add log10 chlorophyll-a
band_vars["HPLCHLA"] = chl_mean_time_series

# Combine spectral bands and chlorophyll
combined_ds = xr.Dataset(
    band_vars
).astype(np.float32)

print("Combined predictor dataset:")
print(combined_ds)


# CONVERT XARRAY DATA TO A DASK DATAFRAME
stacked = (
    combined_ds
    .to_array("variable")
    .transpose("lat", "lon", "variable")
)

# Flatten the latitude-longitude grid into pixels
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

# HANDLE MISSING VALUES
# Calculate global means for each predictor
global_means = df.mean().compute()


def clean_partition(df):
    """
    Remove pixels containing more than 50% missing predictors.
    Fill remaining missing values using global feature means.
    """

    nan_fraction = df.isna().mean(
        axis=1
    )

    df = df[
        nan_fraction <= 0.5
    ]

    return df.fillna(
        global_means
    )


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

# Ensure the expected seasonal columns are available
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.int8(0)


# LOAD TRAINED HAPTOPHYTE SVM MODEL AND SCALER
model = joblib.load(
    "Haptophyte_SVMmodel.joblib"
)

scaler = joblib.load(
    "Haptophyte_SVMscaler.joblib"
)

# Satellite-to-simulation normalization parameters
normalization = joblib.load(
    "Satellite_to_Simulation_MinMaxScaler_2024_AllStations.joblib"
)

print("Model feature names:")
print(scaler.feature_names_in_)

print("\nAvailable dataframe columns:")
print(df.columns)


# DEFINE MODEL INPUT FEATURES
feature_cols = (
    [
        "510", "410", "575", "480",
        "445", "540", "460", "400",
        "495", "615", "470", "425",
        "525", "435", "555", "490"
    ]
    + ["HPLCHLA"]
    + expected_cols
)

df_clean = df[
    feature_cols
]

# DEFINE SATELLITE SPECTRAL BANDS
satellite_band_cols = [
    "510", "410", "575", "480",
    "445", "540", "460", "400",
    "495", "615", "470", "425",
    "525", "435", "555", "490"
]



# CHECK NORMALIZATION PARAMETERS
normalization_wavelengths = np.asarray(
    normalization["wavelengths"],
    dtype=float
)

print(
    "Normalization bands:",
    len(normalization_wavelengths)
)

print(
    "Normalization wavelength range:",
    normalization_wavelengths.min(),
    "to",
    normalization_wavelengths.max(),
    "nm"
)

# NORMALIZE SATELLITE TO SIMULATED REFLECTANCE
def normalize_satellite_to_simulated(X):
    """
    Convert satellite reflectance to the simulated-reflectance
    distribution using the 2024 all-station normalization model.

    Each wavelength is transformed independently using:

        satellite_lower
        satellite_upper
        simulation_lower
        simulation_upper

    The fitted normalization parameters must correspond to the
    model's original training workflow.
    """

    X = X.copy()

    for col in satellite_band_cols:

        wavelength = float(col)

        # Verify that the wavelength is supported
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

        # Find wavelength-specific normalization parameters
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

        # Satellite reflectance limits
        sat_lo = float(
            parameter_record["satellite_lower"]
        )

        sat_hi = float(
            parameter_record["satellite_upper"]
        )

        # Simulated reflectance limits
        sim_lo = float(
            parameter_record["simulation_lower"]
        )

        sim_hi = float(
            parameter_record["simulation_upper"]
        )

        # Validate normalization parameters
        if not np.isfinite(
            [sat_lo, sat_hi, sim_lo, sim_hi]
        ).all():
            raise ValueError(
                f"Invalid normalization parameters "
                f"for {wavelength} nm."
            )

        if np.isclose(
            sat_hi,
            sat_lo
        ):
            raise ValueError(
                f"Zero satellite normalization range "
                f"for {wavelength} nm."
            )

        if np.isclose(
            sim_hi,
            sim_lo
        ):
            raise ValueError(
                f"Zero simulation normalization range "
                f"for {wavelength} nm."
            )

        # Transform satellite values into simulation range
        X[col] = (
            sim_lo
            + (
                (X[col] - sat_lo)
                * (sim_hi - sim_lo)
                / (sat_hi - sat_lo)
            )
        )

    return X


# CREATE DASK PARTITIONS
partitions = df_clean.to_delayed()

# DEFINE PARTITION-WISE PREDICTION FUNCTION
def predict_partition(part):
    """
    Predict haptophyte concentration for one Dask partition.
    """

    # Convert the delayed Dask partition to pandas
    pdf = part.compute()

    if pdf.empty:
        return None

    # Convert predictor values to float32
    X = pdf.astype(
        np.float32
    )

    # Match the exact feature names and order used during training
    X = X[
        scaler.feature_names_in_
    ].copy()

    
    # Step 1: Normalize satellite reflectance
    
    X = normalize_satellite_to_simulated(
        X
    )

    
    # Step 2: Apply the trained model's scaler
    
    X_scaled = scaler.transform(
        X
    )
    # Step 3: Predict haptophyte concentration
    pred = model.predict(
        X_scaled
    )

    return (
        pred,
        pdf.index.to_numpy()
    )

# RUN PREDICTIONS IN PARALLEL

results = Parallel(
    n_jobs=-1,
    backend="loky",
    verbose=10
)(
    delayed(predict_partition)(part)
    for part in partitions
)

# COLLECT PREDICTIONS
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

# Initialize the full spatial array with NaN values
prediction = np.full(
    n_lat * n_lon,
    np.nan,
    dtype=np.float32
)

# Insert predictions at their corresponding pixel indices
prediction[valid_index] = y_pred

# Reshape to latitude-longitude dimensions
prediction = prediction.reshape(
    n_lat,
    n_lon
)

# RELEASE MEMORY
del df
del df_clean
del stacked
del pixels

gc.collect()

# MASK EXTREME PREDICTION VALUES
# Mask predicted values greater than 2
prediction = np.ma.masked_where(
    prediction > 2,
    prediction
)

# Retain the original notebook's lower threshold
prediction = np.ma.masked_where(
    prediction <= -5,
    prediction
)



# PREPARE PREDICTIONS FOR MAPPING
# Predictions remain in log10 space
prediction_plot = np.ma.masked_invalid(
    np.asarray(prediction)
)

# CREATE THE HAPTOPHYTE CONCENTRATION MAP
fig, ax = plt.subplots(
    figsize=(6, 5),
    dpi=150,
    subplot_kw={
        "projection": ccrs.PlateCarree()
    }
)

# Plot predicted haptophyte values
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
# Draw land above the satellite prediction raster
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

# Add borders
ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    linewidth=0.5,
    edgecolor="black",
    zorder=4
)


# SET MAP EXTENT
ax.set_extent(
    [-71, -55, 40, 50],
    crs=ccrs.PlateCarree()
)

# ADD COLORBAR
# The predictions and color scale are in log10 space.
# The tick labels show 10**x, as in the original notebook.
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
    "Haptophytes Concentration (μg/L)",
    fontsize=12
)

# LABEL AXES AND FORMAT THE FIGURE
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
    "PACE_Haptophyte_Concentration_Map.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

print(
    "Haptophyte concentration map generated successfully."
)

