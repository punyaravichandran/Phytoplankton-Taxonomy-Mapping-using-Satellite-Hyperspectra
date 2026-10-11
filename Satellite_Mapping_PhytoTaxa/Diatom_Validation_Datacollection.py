
# ============================================================
# PACE OCI DIATOM VALIDATION
# Daily satellite predictions matched to field station locations
# ============================================================

# IMPORT REQUIRED LIBRARIES
import gc
import earthaccess
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

# LOAD FIELD STATION DATA
points = pd.read_csv("Timeseries_2024_Date_Lat_Lon.csv")

# Keep only useful columns
points = points[
    ["SAMPLE_ID", "Date", "START_LAT", "START_LON"]
].copy()

# Convert dates
points["Date"] = pd.to_datetime(points["Date"])

# Group station observations by date
grouped = points.groupby("Date")

# AUTHENTICATE WITH EARTHACCESS
auth = earthaccess.login()
# PACE DATA PREPROCESSING
def time_from_attr(ds):
    """
    Extract the time coordinate from dataset metadata
    and subset the Scotian Shelf study region.
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

# HANDLE MISSING VALUES
def clean_partition(df):
    """
    Remove rows with more than 50% missing values.
    Fill remaining missing values using column means.
    """

    nan_fraction = df.isna().mean(axis=1)
    df = df[nan_fraction <= 0.5]

    return df.fillna(df.mean())
# LOAD TRAINED DIATOM MODEL AND SCALERS
model = joblib.load("Diatom_SVMmodel.joblib")

scaler = joblib.load("Diatom_SVMscaler.joblib")

# Satellite-to-simulation normalization parameters
normalization = joblib.load(
    "Satellite_to_Simulation_MinMaxScaler_2024_AllStations.joblib"
)

normalization_wavelengths = np.asarray(
    normalization["wavelengths"],
    dtype=float
)

# SATELLITE SPECTRAL BANDS USED BY THE DIATOM SVM
satellite_band_cols = [
    "560", "415", "495", "525", "475", "400",
    "586", "450", "440", "430", "505", "460",
    "485", "540", "650", "425"
]

# NORMALIZE SATELLITE REFLECTANCE TO SIMULATED REFLECTANCE
def normalize_satellite_to_simulated(X):
    """
    Transform satellite reflectance values to the corresponding
    simulated-reflectance range using saved normalization parameters.
    """

    X = X.copy()

    for col in satellite_band_cols:

        wavelength = float(col)

        if wavelength not in normalization_wavelengths:
            raise KeyError(
                f"{wavelength} nm is not available in the "
                "normalization joblib."
            )

        # Find normalization parameters for this wavelength
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
        sat_lo = float(parameter_record["satellite_lower"])
        sat_hi = float(parameter_record["satellite_upper"])

        # Simulated reflectance range
        sim_lo = float(parameter_record["simulation_lower"])
        sim_hi = float(parameter_record["simulation_upper"])

        # Validate normalization parameters
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

# PREDICT DIATOM VALUES FOR EACH DASK PARTITION
def predict_partition(part):
    """
    Apply normalization, trained scaling, and SVM prediction
    to one Dask DataFrame partition.
    """

    # Convert Dask partition to pandas
    pdf = part.compute()

    if pdf.empty:
        return None

    # Preserve feature names and pixel indices
    pdf = pdf.copy()
    pdf.columns = pdf.columns.astype(str)

    pixel_indices = pdf.index.to_numpy()

    # Match the exact feature order used during model training
    training_cols = [
        str(c) for c in scaler.feature_names_in_
    ]

    X = pdf[training_cols].copy()

    # Replace infinite values with NaN
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Impute missing values using partition means,
    # then use zero as a fallback for entirely missing columns
    X = X.fillna(X.mean()).fillna(0)

    X = X.astype(np.float32)

    # Step 1: Satellite-to-simulation reflectance normalization
    X = normalize_satellite_to_simulated(X)

    # Step 2: Apply the trained feature scaler
    X_scaled = scaler.transform(X)

    # Check scaled data
    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "NaN or Inf detected after scaler.transform()."
        )

    # Step 3: SVM prediction
    pred = model.predict(X_scaled)

    return pred, pixel_indices

# DEFINE SEASONS
def get_season(month):
    """
    Assign a season based on the month.
    """

    if month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    elif month in [9, 10, 11]:
        return "Fall"

    else:
        return "Other"

# EXTRACT MEAN AND STANDARD DEVIATION FROM A 3x3 WINDOW
def extract_diatom_3x3(
    lat,
    lon,
    biomass_map,
    query_lats,
    query_lons
):
    """
    Find the nearest satellite grid cell to each field station.
    Calculate the mean and standard deviation within a 3x3
    neighborhood around the nearest cell.
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    biomass_map = np.asarray(biomass_map)

    # Convert 1D coordinates to 2D if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    nrows, ncols = biomass_map.shape

    results = []

    for qlat, qlon in zip(query_lats, query_lons):

        # Find nearest grid cell using squared coordinate distance
        dist2 = (
            (lat2d - qlat) ** 2
            + (lon2d - qlon) ** 2
        )

        iy, ix = np.unravel_index(
            np.argmin(dist2),
            dist2.shape
        )

        # Define the 3x3 window, handling boundaries
        r0 = max(iy - 1, 0)
        r1 = min(iy + 2, nrows)

        c0 = max(ix - 1, 0)
        c1 = min(ix + 2, ncols)

        window = biomass_map[r0:r1, c0:c1]

        # Count valid pixels
        valid = np.isfinite(window)
        n_valid = valid.sum()

        if n_valid > 0:
            mean_val = np.nanmean(window)
            std_val = np.nanstd(window)
        else:
            mean_val = np.nan
            std_val = np.nan

        results.append({
            "lat_query": qlat,
            "lon_query": qlon,
            "mean_diatom": mean_val,
            "std_diatom": std_val,
            "valid_pixels": n_valid
        })

    return pd.DataFrame(results)

# PROCESS PACE DATA FOR ONE DAY
def process_one_day(date, stations_today):
    """
    Retrieve daily PACE reflectance and chlorophyll data,
    predict diatom values, and extract satellite estimates
    at field station locations.
    """

    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    tspan = (date_str, date_str)

    print(f"Searching PACE data for {date_str}")
    # Search for PACE OCI surface reflectance data
    rrs_results = earthaccess.search_data(
        short_name="PACE_OCI_L3M_SFREFL",
        temporal=tspan,
        granule_name="*DAY.*.4km.*"
    )
    # Search for PACE OCI biogeochemical data
    chl_results = earthaccess.search_data(
        short_name="PACE_OCI_L3M_BGC",
        temporal=tspan,
        granule_name="*DAY.*.4km.*"
    )

    # Check data availability
    if not rrs_results:
        print(f"No reflectance data found for {date_str}")
        return None

    if not chl_results:
        print(f"No chlorophyll data found for {date_str}")
        return None
    # Open PACE datasets
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

    # Extract month and season
    Date = pd.to_datetime(dataset1.date.values[0])

    month = Date.month
    season = get_season(month)

    # Rename chlorophyll variable
    dataset2 = dataset2.rename({
        "chlor_a": "HPLCHLA"
    })

    # Rechunk after loading metadata
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
    # Extract reflectance and chlorophyll variables
    rrs = dataset1["rhos"]

    chl_raw = dataset2["HPLCHLA"]

    # Log10-transform positive chlorophyll values
    chl = np.log10(
        chl_raw.where(chl_raw > 0)
    )

    # Keep positive reflectance values
    rrs = rrs.where(rrs > 0)

    # Average over time
    rrs_visible = rrs.sel(
        wavelength=slice(400, 701)
    )

    rrs_mean_time_series = rrs_visible.mean(
        dim="date"
    )

    chl_mean_time_series = chl.mean(
        dim="date"
    )
    # Select the spectral bands used by the trained model
    selected_wavelengths = [
        560, 415, 495, 525, 475, 400,
        586, 450, 440, 430, 505, 460,
        485, 540, 650, 425
    ]

    refl_subset = rrs_mean_time_series.sel(
        wavelength=selected_wavelengths
    )

    # Align reflectance and chlorophyll spatially
    refl_subset, chl_mean_time_series = xr.align(
        refl_subset,
        chl_mean_time_series,
        join="inner"
    )
    # Build predictor dataset
    band_vars = {
        str(int(w)): (
            refl_subset
            .sel(wavelength=w)
            .drop_vars("wavelength")
        )
        for w in refl_subset.wavelength.values
    }

    # Add log10 chlorophyll
    band_vars["HPLCHLA"] = chl_mean_time_series

    # Combine into one dataset
    combined_ds = xr.Dataset(
        band_vars
    ).astype(np.float32)
    # Convert the spatial dataset into a Dask DataFrame
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

    # Remove rows with excessive missing values
    df = df.map_partitions(
        clean_partition
    )
    # Add seasonal features
    df["Season"] = season
    df["Season"] = df["Season"].astype("category")

    df["Season_Spring"] = np.int8(
        season == "Spring"
    )

    df["Season_Summer"] = np.int8(
        season == "Summer"
    )

    df["Season_Fall"] = np.int8(
        season == "Fall"
    )
    # Match exact training features
    training_cols = [
        str(c) for c in scaler.feature_names_in_
    ]

    missing_cols = [
        col
        for col in training_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing model features: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Preserve exact feature order used during training
    df_clean = df[training_cols]

    # Convert Dask partitions to delayed objects
    partitions = df_clean.to_delayed()
    # Predict in parallel
    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10
    )(
        delayed(predict_partition)(part)
        for part in partitions
    )
    # Merge partition predictions
    predictions = []
    indices = []

    for result in results:

        if result is None:
            continue

        pred, idx = result

        predictions.append(pred)
        indices.append(idx)

    if len(predictions) == 0:
        print(f"No valid pixels for {date_str}")
        return None

    y_pred = np.concatenate(predictions)
    valid_index = np.concatenate(indices)
    # Create the spatial prediction map
    n_lat = combined_ds.sizes["lat"]
    n_lon = combined_ds.sizes["lon"]

    prediction = np.full(
        n_lat * n_lon,
        np.nan,
        dtype=np.float32
    )

    prediction[valid_index] = y_pred

    prediction = prediction.reshape(
        n_lat,
        n_lon
    )

    biomass_map = prediction

    
    # Extract predictions around field stations
    
    lat = combined_ds.lat
    lon = combined_ds.lon

    nearest = extract_diatom_3x3(
        lat=lat,
        lon=lon,
        biomass_map=biomass_map,
        query_lats=stations_today["START_LAT"].values,
        query_lons=stations_today["START_LON"].values
    )

    output = stations_today.reset_index(
        drop=True
    ).copy()

    output["Predicted_Diatoms"] = (
        nearest["mean_diatom"].values
    )

    output["STD_Diatoms"] = (
        nearest["std_diatom"].values
    )

    # Release memory
    
    del df
    del df_clean
    del stacked
    del pixels
    del combined_ds
    del dataset1
    del dataset2
    del rrs
    del chl

    gc.collect()

    return output


# PROCESS ALL FIELD OBSERVATION DATES
all_results = []

for date, stations_today in grouped:

    print(f"\nProcessing {date.date()}")

    day_result = process_one_day(
        date,
        stations_today
    )

    if day_result is not None:
        all_results.append(day_result)

# COMBINE RESULTS AND SAVE CSV
if all_results:

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    final_results.to_csv(
        "Predicted_Diatom_2024.csv",
        index=False
    )

    print(
        f"Saved {len(final_results)} predictions "
        "to Predicted_Diatom_2024.csv"
    )

else:

    print(
        "No predictions generated. "
        "Check data availability and preprocessing."
    )

