"""
SPA feature selection and SVR for Dinoflagellates biomass estimation
Author: Punya
"""

## Import Libraries and Load Data
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import optuna
import shap
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedKFold,
)
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    make_scorer,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

## Load and Merge Datasets
# Load spectral reflectance and pigment absorption datasets
SAMRef_data = pd.read_csv(
    r"/content/Reflectance_SAM_2019-2024_cdom160.csv"
)
Absorption_data = pd.read_csv(
    r"/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)
# Rename the sample ID column
SAMRef_data.rename(
    columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},
    inplace=True,
)
# Merge the two dataframes using SAMPLE_ID
Final_df = pd.merge(
    SAMRef_data,
    Absorption_data,
    on="SAMPLE_ID",
)
# Convert dates to pandas datetime format
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"])
# Extract year, month, and day
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day

## Filter Data by Depth
# Retain samples collected at depths less than 20 metres
Filtered_data = Final_df[Final_df["DEPTH"] < 20]
Filtered_data.head()

## Calculate Dinoflagellate Percentage Biomass
# Calculate dinoflagellate biomass as a percentage of HPLC chlorophyll-a
taxa_biomass = Filtered_data["dino1"].div(
    Filtered_data["HPLCHLA"],
    axis=0,
)
taxa_percent = (taxa_biomass * 100).rename("dino_percent")
# Add percentage biomass to the reflectance dataset
Reflection_taxa = pd.concat(
    [Filtered_data, taxa_percent],
    axis=1,
)

## Seasonal Analysis
# Create a month-indexed dataframe
HL_data_month = Reflection_taxa.set_index(
    Filtered_data["SAMPLE_DATE"].dt.month
)
HL_data_month.index.name = "Months"
# Replace month numbers with month names
HL_data_month = HL_data_month.rename(
    index={
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
)
HL_data_month = HL_data_month.reset_index()
# Assign seasons based on month names
def get_season(month):
    if month in ["March", "April", "May"]:
        return "Spring"
    elif month in ["June", "July", "August"]:
        return "Summer"
    elif month in ["September", "October", "November"]:
        return "Fall"
    else:
        return "Other"

HL_data_month["Season"] = HL_data_month["Months"].apply(get_season)

## Clean Data and Remove Dinoflagellate Outliers
# Remove rows containing missing values
Filtered_data_cleaned = Filtered_data.dropna().copy()
# Define percentile thresholds
lower_percentile = 2
upper_percentile = 98
# Consider only positive dinoflagellate concentrations
diatom_values_for_percentile = Filtered_data_cleaned.loc[
    Filtered_data_cleaned["dino1"] > 0,
    "dino1",
]

if not diatom_values_for_percentile.empty:
    lower_bound = np.percentile(
        diatom_values_for_percentile,
        lower_percentile,
    )
    upper_bound = np.percentile(
        diatom_values_for_percentile,
        upper_percentile,
    )
    original_num_rows = Filtered_data_cleaned.shape[0]
    # Retain positive concentrations within the percentile range
    Filtered_data_cleaned = Filtered_data_cleaned[
        (Filtered_data_cleaned["dino1"] >= lower_bound)
        & (Filtered_data_cleaned["dino1"] <= upper_bound)
        & (Filtered_data_cleaned["dino1"] > 0)
    ].copy()
    print(
        "Removed "
        f"{original_num_rows - Filtered_data_cleaned.shape[0]} "
        "rows due to dinoflagellate percentile filtering."
    )
    print(
        "New number of rows in Filtered_data_cleaned: "
        f"{Filtered_data_cleaned.shape[0]}"
    )
else:
    print(
        "Warning: The 'dino1' column is empty or contains only "
        "non-positive values. Percentile filtering was not applied."
    )

## Filter-Based Spectral Band Reduction
# Select spectral reflectance bands
spectral_bands = [
    "Refl400nm", "Refl405nm", "Refl410nm", "Refl415nm",
    "Refl420nm", "Refl425nm", "Refl430nm", "Refl435nm",
    "Refl440nm", "Refl445nm", "Refl450nm", "Refl455nm",
    "Refl460nm", "Refl465nm", "Refl470nm", "Refl475nm",
    "Refl480nm", "Refl485nm", "Refl490nm", "Refl495nm",
    "Refl500nm", "Refl505nm", "Refl510nm", "Refl515nm",
    "Refl520nm", "Refl525nm", "Refl530nm", "Refl535nm",
    "Refl540nm", "Refl545nm", "Refl550nm", "Refl555nm",
    "Refl560nm", "Refl565nm", "Refl570nm", "Refl575nm",
    "Refl580nm", "Refl586nm", "Refl615nm", "Refl620nm",
    "Refl625nm", "Refl630nm", "Refl635nm", "Refl640nm",
    "Refl642nm", "Refl645nm", "Refl647nm", "Refl650nm",
    "Refl652nm", "Refl655nm", "Refl657nm", "Refl660nm",
    "Refl662nm", "Refl665nm", "Refl667nm", "Refl670nm",
    "Refl672nm", "Refl675nm", "Refl677nm", "Refl679nm",
    "Refl682nm", "Refl697nm", "Refl699nm",
]
Filtered_data_cleaned1 = Filtered_data_cleaned[
    spectral_bands
].copy()

# Rename reflectance columns, for example Refl400nm -> 400
column_mapping = {
    col: col.replace("Refl", "").replace("nm", "")
    for col in Filtered_data_cleaned1.columns
}
Filtered_data_cleaned1 = Filtered_data_cleaned1.rename(
    columns=column_mapping,
)

## Define Features and Log-Transformed Target
# Define spectral reflectance features
X = Filtered_data_cleaned1.copy()
# Retrieve the dinoflagellate concentration using matching indices
diatom_values = Filtered_data.dropna().loc[
    X.index,
    "dino1",
]
# Retain non-negative values
positive_diatom_values = diatom_values[
    diatom_values >= 0
]
# Apply log10 transformation
y_candidate = np.log10(positive_diatom_values).fillna(0)
# Retain finite target values
valid_indices = y_candidate[
    np.isfinite(y_candidate)
].index
# Align features and target
X = X.loc[valid_indices].copy()
y = y_candidate.loc[valid_indices].copy()

## Successive Projections Algorithm (SPA)
# Prepare features and target for SPA
X_selection = Filtered_data_cleaned1.copy()
y_selection = y.copy()
# SPA configuration
MAX_BANDS_SPA = 25
STEP_SPA = 1

def spa_selection(X_data, k, y_data=None):
    X_matrix = X_data.values
    n_samples, n_features = X_matrix.shape
    selected_indices = []
    available_indices = list(range(n_features))
    # Select the initial feature
    if y_data is not None:
        abs_correlations = X_data.corrwith(
            y_data.loc[X_data.index]
        ).abs()
        initial_idx = X_data.columns.get_loc(
            abs_correlations.idxmax()
        )
    else:
        initial_idx = np.argmax(
            np.linalg.norm(X_matrix, axis=0)
        )
    selected_indices.append(initial_idx)
    available_indices.remove(initial_idx)
    # Iteratively select additional features
    for _ in range(k - 1):

        if not available_indices:
            break

        A = X_matrix[:, selected_indices]
        max_residual_norm = -1
        next_feature_idx = -1

        for j in available_indices:
            current_feature = X_matrix[:, j]
            try:
                beta = np.linalg.lstsq(
                    A,
                    current_feature,
                    rcond=None,
                )[0]
                projected_vector = A @ beta
                residual_vector = (
                    current_feature - projected_vector
                )
                residual_norm = np.linalg.norm(
                    residual_vector
                )
            except np.linalg.LinAlgError:
                residual_norm = 0.0

            if residual_norm > max_residual_norm:
                max_residual_norm = residual_norm
                next_feature_idx = j

        if next_feature_idx != -1:
            selected_indices.append(next_feature_idx)
            available_indices.remove(next_feature_idx)
        else:
            break
    return X_data.columns[selected_indices].tolist()

## Optimize the Number of SPA Bands Using SVR Cross-Validation
def optimize_spa_bands(
    X,
    y,
    max_bands,
    step,
    SVR_params,
):
    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []

    for k in range(5, max_bands + 1, step):
        if k > X.shape[1]:
            break

        print(f"Evaluating SPA with {k} bands...")

        try:
            current_bands = spa_selection(
                X,
                k,
                y,
            )
        except Exception as e:
            print(
                f"SPA selection failed for k={k}: {e}"
            )
            continue

        if not current_bands:
            print(
                f"No bands selected for k={k}. Skipping."
            )
            continue
        X_k = X[current_bands]
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(**SVR_params)),
            ]
        )
        r2_scores = cross_val_score(
            pipeline,
            X_k,
            y,
            cv=5,
            scoring=make_scorer(r2_score),
            n_jobs=-1,
        )
        mean_r2 = np.mean(r2_scores)
        print(
            f"Mean R² for {k} bands: {mean_r2:.4f}"
        )
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = current_bands

    return optimal_bands, optimal_k, best_r2

# Generic SVR parameters for SPA optimization
generic_svr_params = {
    "kernel": "rbf",
    "C": 10,
    "epsilon": 0.1,
    "gamma": "scale",
}
print(
    "Starting SPA + SVR cross-validation "
    "for optimal band identification..."
)
optimal_bands_spa, optimal_k_spa, best_r2_spa_cv = (
    optimize_spa_bands(
        X_selection,
        y_selection,
        MAX_BANDS_SPA,
        STEP_SPA,
        generic_svr_params,
    )
)
print(
    "\nOptimal number of bands selected by SPA + SVR CV:",
    optimal_k_spa,
)
print(
    f"Best R² achieved during optimization: "
    f"{best_r2_spa_cv:.4f}"
)
print("Optimal bands:", optimal_bands_spa)

# Store selected bands
selected_bands = {}
importance_scores = {}
selected_bands["SPA + SVR CV"] = pd.Index(
    optimal_bands_spa
)

## Evaluate the Selected Spectral Bands
results = {}
method_name = "SPA + SVR CV"
bands = selected_bands[method_name]
if not bands.empty:
    X_sel = X_selection[bands]
    scaler = StandardScaler()
    X_sel_scaled = scaler.fit_transform(X_sel)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel_scaled,
        y_selection,
        test_size=0.2,
        random_state=42,
    )
    model = SVR(**generic_svr_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[method_name] = r2_score(
        y_test,
        y_pred,
    )
else:
    results[method_name] = np.nan
print(
    "\nR² Scores for SPA + SVR Feature Selection Method:"
)

for method, score in results.items():
    print(f"{method}: {score:.4f}")
    print(
        f"Selected {len(selected_bands[method])} bands.\n"
    )

# Store the selected bands
overlap_bands = selected_bands
print(overlap_bands)

## Prepare Features and Target for Machine Learning
# Retain the selected spectral bands
X = X[overlap_bands["SPA + SVR CV"]].copy()
# Add log-transformed chlorophyll-a as a predictor
X["HPLCHLA"] = np.log10(
    Filtered_data_cleaned["HPLCHLA"]
).fillna(0)
# Filter target values
filtered_indices = y[y >= -8].index
X_filtered = X.loc[filtered_indices].copy()
y_filtered = y.loc[filtered_indices]

# Assign seasons using numeric month values
def get_season_from_month(month_num):
    if month_num in [3, 4, 5]:
        return "Spring"
    elif month_num in [6, 7, 8]:
        return "Summer"
    elif month_num in [9, 10, 11]:
        return "Fall"
    else:
        return "Other"

filtered_data_cleaned_seasons = (
    Filtered_data_cleaned["MONTH"].apply(
        get_season_from_month
    )
)
# Match season data to feature indices
season_data = filtered_data_cleaned_seasons.loc[
    X_filtered.index
]
# One-hot encode seasons
season_dummies = pd.get_dummies(
    season_data,
    prefix="Season",
    dtype=int,
)
# Add seasonal features
X_filtered = pd.concat(
    [X_filtered, season_dummies],
    axis=1,
)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered,
    y_filtered,
    test_size=0.2,
    random_state=42,
)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_test_original_index = y_test.index
print(
    f"Number of training samples: {X_train.shape[0]}"
)
print(
    f"Number of testing samples: {X_test.shape[0]}"
)

## Hyperparameter Tuning Using Optuna
def objective(trial):
    C = trial.suggest_float(
        "C",
        1e-2,
        1e3,
        log=True,
    )
    epsilon = trial.suggest_float(
        "epsilon",
        1e-3,
        1.0,
        log=True,
    )
    gamma = trial.suggest_categorical(
        "gamma",
        ["scale", "auto"],
    )
    kernel = trial.suggest_categorical(
        "kernel",
        ["rbf", "poly", "sigmoid"],
    )
    degree = (
        trial.suggest_int("degree", 2, 5)
        if kernel == "poly"
        else 3
    )

    # Create SVR model
    svr = SVR(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel,
        degree=degree,
    )
    # Standardization and SVR pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        svr,
    )
    # Repeated five-fold cross-validation
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=make_scorer(r2_score),
        n_jobs=-1,
    )
    return scores.mean()

# Create Optuna study
study = optuna.create_study(
    direction="maximize"
)
study.optimize(
    objective,
    n_trials=150,
)
# Display the best trial
print("Best trial:")
trial = study.best_trial
print(f"R² Score: {trial.value:.4f}")
print("Best parameters:")

for key, value in trial.params.items():
    print(f"  {key}: {value}")

## Train and Save the Best SVR Model
# Retrieve optimized parameters
best_params = study.best_trial.params
# Create the final SVR model
best_model = make_pipeline(
    SVR(
        C=best_params["C"],
        epsilon=best_params["epsilon"],
        gamma=best_params["gamma"],
        kernel=best_params["kernel"],
        degree=best_params.get("degree", 3),
    )
)
# Train the model
best_model.fit(X_train, y_train)
# Predict test data
y_pred = best_model.predict(X_test)
# Save the model and scaler
joblib.dump(
    best_model,
    "Dinoflagellate_SVMmodel.joblib",
)
joblib.dump(
    scaler,
    "Dinoflagellate_SVMscaler.joblib",
)

## Model Evaluation
# Calculate model performance
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(
    y_test,
    y_pred,
)
print("R²:", r2)
print("RMSE:", rmse)

## Actual Versus Predicted Dinoflagellate Concentration
plt.rcParams["figure.figsize"] = (4.5, 5)
# Convert log-transformed values to the original scale
plot_df = pd.DataFrame(
    {
        "Actual Dino Concentration": 10**y_test,
        "SVR Model Dino Concentration": 10**y_pred,
    },
    index=y_test_original_index,
)
# Add seasonal and percentage biomass information
plot_df = plot_df.merge(
    HL_data_month[["Season", "dino_percent"]],
    left_index=True,
    right_index=True,
    how="left",
)
# Define season order
desired_season_order = [
    "Spring",
    "Summer",
    "Fall",
]
plot_df["Season"] = pd.Categorical(
    plot_df["Season"],
    categories=desired_season_order,
    ordered=True,
)
season_map = {
    code: season
    for code, season in enumerate(
        plot_df["Season"].cat.categories
    )
}
# Point-size offset
min_size_offset = 5
# Create scatter plot
scatter = plt.scatter(
    plot_df["Actual Dino Concentration"],
    plot_df["SVR Model Dino Concentration"],
    c=plot_df["Season"].cat.codes,
    s=plot_df["dino_percent"] + min_size_offset,
    cmap="rainbow",
    alpha=0.5,
)
# Determine axis limits
all_values = pd.concat(
    [
        plot_df["Actual Dino Concentration"],
        plot_df["SVR Model Dino Concentration"],
    ]
)
min_val = all_values.min() - 0.2
max_val = all_values.max()
# Add 1:1 reference line
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "g--",
)
plt.xlabel(
    "Actual Concentration (μg/L)",
    fontsize=12,
)
plt.ylabel(
    "Model Concentration (μg/L)",
    fontsize=12,
)
plt.title(
    "Dinoflagellates",
    fontsize=12,
)
plt.grid(
    True,
    linestyle="--",
    alpha=0.6,
)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
# Add evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(
    y_test,
    y_pred,
)
rmse = np.sqrt(mse)
plt.text(
    0.60,
    0.97,
    f"R²: {r2:.2f}\nRMSE: {rmse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=13,
    verticalalignment="top",
)
# Add season colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map.keys()))
cbar.set_ticklabels(list(season_map.values()))
# Add percentage-biomass size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset,
]:
    plt.scatter(
        [],
        [],
        c="gray",
        alpha=0.6,
        s=size,
        label=f"{size - min_size_offset}%",
    )
plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title="Dino Percentage",
)
plt.show()

## Model Diagnostics: Residual Plot
# Calculate residuals on the original concentration scale
residuals_normal = 10**y_test - 10**y_pred
residual_plot_df = pd.DataFrame(
    {
        "Predicted Dino Concentration": 10**y_pred,
        "Residuals": residuals_normal,
    },
    index=y_test_original_index,
)
# Add season and percentage biomass information
residual_plot_df = residual_plot_df.merge(
    HL_data_month[["Season", "dino_percent"]],
    left_index=True,
    right_index=True,
    how="left",
)
# Define season order
desired_season_order_residual = [
    "Spring",
    "Summer",
    "Fall",
]
residual_plot_df["Season"] = pd.Categorical(
    residual_plot_df["Season"],
    categories=desired_season_order_residual,
    ordered=True,
)
season_map_residual = {
    code: season
    for code, season in enumerate(
        residual_plot_df["Season"].cat.categories
    )
}
min_size_offset = 5
plt.figure(figsize=(5, 3))
# Create residual scatter plot
scatter = plt.scatter(
    residual_plot_df["Predicted Dino Concentration"],
    residual_plot_df["Residuals"],
    c=residual_plot_df["Season"].cat.codes,
    s=residual_plot_df["dino_percent"] + min_size_offset,
    cmap="rainbow",
    alpha=0.5,
)
# Reference line at zero residual
plt.axhline(
    0,
    color="red",
    linestyle="--",
)
plt.xlabel(
    "Predicted Biomass (μg/L)",
    fontsize=12,
)
plt.ylabel(
    "Residuals (μg/L)",
    fontsize=12,
)
plt.title("Dinoflagellates")
plt.grid(True)
# Add season colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(
    list(season_map_residual.keys())
)
cbar.set_ticklabels(
    list(season_map_residual.values())
)
cbar.set_label("Season")
# Add percentage-biomass size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset,
]:
    plt.scatter(
        [],
        [],
        c="gray",
        alpha=0.6,
        s=size,
        label=f"{size - min_size_offset}%",
    )
plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title="Dino Percentage",
)
plt.show()

## SHAP Feature Interpretation
# Create an independent masker
masker = shap.maskers.Independent(
    X_train,
    max_samples=X_train.shape[0],
)
# Explain predictions from the trained SVR model
explainer = shap.Explainer(
    best_model.named_steps["svr"].predict,
    masker,
)
shap_values = explainer(X_test)
# Retrieve feature names
feature_names = list(X_filtered.columns)
# Rename HPLCHLA to CHL for display
if "HPLCHLA" in feature_names:
    hplchla_index = feature_names.index("HPLCHLA")
    feature_names[hplchla_index] = "CHL"
shap_values.feature_names = feature_names
# Plot the SHAP beeswarm summary
plt.rcParams["figure.figsize"] = (3, 4)
shap.plots.beeswarm(
    shap_values,
    max_display=10,
    order=shap_values.abs.mean(0),
)
