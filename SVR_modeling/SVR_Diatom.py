"""
SPA feature selection and SVR for Diatom biomass estimation
from reflectance, chlorophyll and season information
Author: Punya
"""

# Import packages and load data
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    make_scorer,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import optuna
import shap

# Update these paths if the CSV files are stored somewhere other than /content.
SAMRef_data = pd.read_csv(
    r"/content/Reflectance_SAM_2019-2024_cdom160.csv"
)
Absorption_data = pd.read_csv(
    r"/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)

# Rename the column and merge the two dataframes.
SAMRef_data.rename(columns={"isdata$SAMPLE_ID": "SAMPLE_ID"}, inplace=True)
Final_df = pd.merge(SAMRef_data, Absorption_data, on="SAMPLE_ID")
# Prepare date and depth-filtered data
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"])
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day
# Filter the data to a depth of less than 20 meters.
Filtered_data = Final_df[Final_df["DEPTH"] < 20]

# Calculate diatom percentage biomass
# Percentage biomass = diatom / total biomass * 100.
taxa_biomass = Filtered_data["diatom"].div(Filtered_data["HPLCHLA"], axis=0)
taxa_percent = (taxa_biomass * 100).rename("diatom_percent")
Reflection_taxa = pd.concat([Filtered_data, taxa_percent], axis=1)

# Seasonal analysis
HL_data_month = Reflection_taxa.set_index(Filtered_data["SAMPLE_DATE"].dt.month)
HL_data_month.index.name = "Months"
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

def get_season(month):
    """Return season labels from month names."""
    if month in ["March", "April", "May"]:
        return "Spring"
    if month in ["June", "July", "August"]:
        return "Summer"
    if month in ["September", "October", "November"]:
        return "Fall"
    return "Other"
HL_data_month["Season"] = HL_data_month["Months"].apply(get_season)

# Clean data and filter diatom outliers
Filtered_data_cleaned = Filtered_data.dropna().copy()
lower_percentile = 2
upper_percentile = 98
diatom_values_for_percentile = Filtered_data_cleaned.loc[
    Filtered_data_cleaned["diatom"] > 0, "diatom"
]

if not diatom_values_for_percentile.empty:
    lower_bound = np.percentile(diatom_values_for_percentile, lower_percentile)
    upper_bound = np.percentile(diatom_values_for_percentile, upper_percentile)
    original_num_rows = Filtered_data_cleaned.shape[0]
    Filtered_data_cleaned = Filtered_data_cleaned[
        (Filtered_data_cleaned["diatom"] >= lower_bound)
        & (Filtered_data_cleaned["diatom"] <= upper_bound)
        & (Filtered_data_cleaned["diatom"] > 0)
    ].copy()
    print(
        "Removed "
        f"{original_num_rows - Filtered_data_cleaned.shape[0]} rows due to "
        "'diatom' percentile filtering."
    )
    print(
        "New number of rows in Filtered_data_cleaned: "
        f"{Filtered_data_cleaned.shape[0]}"
    )
else:
    print(
        "Warning: 'diatom' column is empty or contains only non-positive values "
        "after initial filtering. No percentile filtering applied."
    )

# Filter-based spectral band reduction
Filtered_data_cleaned1 = Filtered_data_cleaned[
    [
        "Refl400nm", "Refl405nm", "Refl410nm", "Refl415nm", "Refl420nm",
        "Refl425nm", "Refl430nm", "Refl435nm", "Refl440nm", "Refl445nm",
        "Refl450nm", "Refl455nm", "Refl460nm", "Refl465nm", "Refl470nm",
        "Refl475nm", "Refl480nm", "Refl485nm", "Refl490nm", "Refl495nm",
        "Refl500nm", "Refl505nm", "Refl510nm", "Refl515nm", "Refl520nm",
        "Refl525nm", "Refl530nm", "Refl535nm", "Refl540nm", "Refl545nm",
        "Refl550nm", "Refl555nm", "Refl560nm", "Refl565nm", "Refl570nm",
        "Refl575nm", "Refl580nm", "Refl586nm", "Refl615nm", "Refl620nm",
        "Refl625nm", "Refl630nm", "Refl635nm", "Refl640nm", "Refl642nm",
        "Refl645nm", "Refl647nm", "Refl650nm", "Refl652nm", "Refl655nm",
        "Refl657nm", "Refl660nm", "Refl662nm", "Refl665nm", "Refl667nm",
        "Refl670nm", "Refl672nm", "Refl675nm", "Refl677nm", "Refl679nm",
        "Refl682nm", "Refl697nm", "Refl699nm",
    ]
].copy()

# Normalization with 440 nm absorption was commented out in the supplied code.
column_mapping = {
    col: col.replace("Refl", "").replace("nm", "")
    for col in Filtered_data_cleaned1.columns
}
Filtered_data_cleaned1 = Filtered_data_cleaned1.rename(columns=column_mapping)

# Define X and log-transformed y
X = Filtered_data_cleaned1.copy()
diatom_values = Filtered_data.dropna().loc[X.index, "diatom"]
# Keep non-negative values, as in the supplied code.
# Non-finite log values are removed in the next step.
positive_diatom_values = diatom_values[diatom_values >= 0]
y_candidate = np.log10(positive_diatom_values).fillna(0)
valid_indices = y_candidate[np.isfinite(y_candidate)].index
X = X.loc[valid_indices].copy()
y = y_candidate.loc[valid_indices].copy()

# Successive Projections Algorithm (SPA) + SVR cross-validation
X_selection = Filtered_data_cleaned1.copy()
y_selection = y.copy()
MAX_BANDS_SPA = 25
STEP_SPA = 1

def spa_selection(X_data, k, y_data=None):
    X_matrix = X_data.values
    _, n_features = X_matrix.shape

    selected_indices = []
    available_indices = list(range(n_features))

    if y_data is not None:
        abs_correlations = X_data.corrwith(y_data.loc[X_data.index]).abs()
        initial_idx = X_data.columns.get_loc(abs_correlations.idxmax())
    else:
        initial_idx = np.argmax(np.linalg.norm(X_matrix, axis=0))

    selected_indices.append(initial_idx)
    available_indices.remove(initial_idx)

    for _ in range(k - 1):
        if not available_indices:
            break

        A = X_matrix[:, selected_indices]
        max_residual_norm = -1
        next_feature_idx = -1

        for j in available_indices:
            current_feature = X_matrix[:, j]

            try:
                beta = np.linalg.lstsq(A, current_feature, rcond=None)[0]
                projected_vector = A @ beta
                residual_vector = current_feature - projected_vector
                residual_norm = np.linalg.norm(residual_vector)
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


def optimize_spa_bands(X, y, max_bands, step, SVR_params):
    """Choose the number of SPA bands with the highest mean CV R²."""
    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []

    for k in range(5, max_bands + 1, step):
        if k > X.shape[1]:
            break

        print(f"  Evaluating SPA with {k} bands...")

        try:
            current_bands = spa_selection(X, k, y)
        except Exception as exc:
            print(f"    SPA selection failed for k={k}: {exc}")
            continue

        if not current_bands:
            print(f"    No bands selected for k={k}, skipping.")
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
        print(f"    Mean R2 for {k} bands: {mean_r2:.4f}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = current_bands

    return optimal_bands, optimal_k, best_r2

generic_svr_params = {
    "kernel": "rbf",
    "C": 10,
    "epsilon": 0.1,
    "gamma": "scale",
}
print("Starting SPA + SVR Cross-Validation for optimal band identification...")
optimal_bands_spa, optimal_k_spa, best_r2_spa_cv = optimize_spa_bands(
    X_selection,
    y_selection,
    MAX_BANDS_SPA,
    STEP_SPA,
    generic_svr_params,
)
print(f"\nOptimal number of bands selected by SPA + SVR CV: {optimal_k_spa}")
print(f"Best R2 achieved during optimization: {best_r2_spa_cv:.4f}")
print(f"Optimal bands: {optimal_bands_spa}")
selected_bands = {}
importance_scores = {}
selected_bands["SPA + SVR CV"] = pd.Index(optimal_bands_spa)

# Evaluate the selected bands
results = {}
method_name = "SPA + SVR CV"
bands = selected_bands[method_name]

if not bands.empty:
    X_sel = X_selection[bands]
    scaler = StandardScaler()
    X_sel_scaled = scaler.fit_transform(X_sel)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel_scaled,
        y_selection,
        test_size=0.2,
        random_state=42,
    )

    model = SVR(**generic_svr_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[method_name] = r2_score(y_test, y_pred)
else:
    results[method_name] = np.nan
print("\nR2 Scores for SPA + SVR Feature Selection Method:")

for method, score in results.items():
    print(f"{method}: {score:.4f}")
    print(f"-> Selected {len(selected_bands[method])} bands.\n")

overlap_bands = selected_bands  # Kept as in the supplied code.
print(overlap_bands)

# Prepare model features, including chlorophyll and season
X = X[overlap_bands["SPA + SVR CV"]].copy()
X["HPLCHLA"] = np.log10(Filtered_data_cleaned["HPLCHLA"]).fillna(0)
y = y
filtered_indices = y[y >= -8].index
X_filtered = X.loc[filtered_indices].copy()
y_filtered = y.loc[filtered_indices]

def get_season_from_month(month_num):
    """Return season labels from numeric month values."""
    if month_num in [3, 4, 5]:
        return "Spring"
    if month_num in [6, 7, 8]:
        return "Summer"
    if month_num in [9, 10, 11]:
        return "Fall"
    return "Other"

filtered_data_cleaned_seasons = Filtered_data_cleaned["MONTH"].apply(
    get_season_from_month
)
season_data = filtered_data_cleaned_seasons.loc[X_filtered.index]
season_dummies = pd.get_dummies(season_data, prefix="Season", dtype=int)
X_filtered = pd.concat([X_filtered, season_dummies], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered,
    y_filtered,
    test_size=0.2,
    random_state=42,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_test_original_index = y_test.index
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of testing samples: {X_test.shape[0]}")

# Hyperparameter tuning with Optuna
def objective(trial):
    """Maximize mean R² across repeated five-fold cross-validation."""
    C = trial.suggest_float("C", 1e-2, 1e3, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
    svr = SVR(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel,
        degree=degree,
    )

    pipeline = make_pipeline(StandardScaler(), svr)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=r2_scorer,
        n_jobs=-1,
    )
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)
print("Best trial:")
trial = study.best_trial
print(f"  R2 Score: {trial.value:.4f}")
print("  Params:")

for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Fit best SVR model, save model and scaler
best_params = trial.params
best_model = make_pipeline(
    SVR(
        C=best_params["C"],
        epsilon=best_params["epsilon"],
        gamma=best_params["gamma"],
        kernel=best_params["kernel"],
        degree=best_params.get("degree", 3),
    )
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
joblib.dump(best_model, "Diatom_SVMmodel.joblib")
joblib.dump(scaler, "Diatom_SVMscaler.joblib")

# Model evaluation
filtered_y_test = y_test
filtered_y_pred = y_pred
print("R²:", r2_score(filtered_y_test, filtered_y_pred))
print("RMSE:", root_mean_squared_error(filtered_y_test, filtered_y_pred))

# Actual versus predicted concentration visualization
plt.rcParams["figure.figsize"] = (4.5, 5)
plot_df = pd.DataFrame(
    {
        "Actual Diatom Concentration": 10**y_test,
        "SVR Model Diatom Concentration": 10**y_pred,
    },
    index=y_test_original_index,
)
plot_df = plot_df.merge(
    HL_data_month[["Season", "diatom_percent"]],
    left_index=True,
    right_index=True,
    how="left",
)
desired_season_order = ["Spring", "Summer", "Fall"]
plot_df["Season"] = pd.Categorical(
    plot_df["Season"],
    categories=desired_season_order,
    ordered=True,
)
season_map = {
    code: season
    for code, season in enumerate(plot_df["Season"].cat.categories)
}
min_size_offset = 5
scatter = plt.scatter(
    plot_df["Actual Diatom Concentration"],
    plot_df["SVR Model Diatom Concentration"],
    c=plot_df["Season"].astype("category").cat.codes,
    s=plot_df["diatom_percent"] + min_size_offset,
    cmap="rainbow",
    alpha=0.5,
)
all_values = pd.concat(
    [
        plot_df["Actual Diatom Concentration"],
        plot_df["SVR Model Diatom Concentration"],
    ]
)
min_val = all_values.min() - 0.1
max_val = all_values.max()
plt.plot([min_val, max_val], [min_val, max_val], "g--")
plt.xlabel("Actual Concentration (μg/L)", fontsize=12)
plt.ylabel("Model Concentration (μg/L)", fontsize=12)
plt.title("Diatoms", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
plt.text(
    0.58,
    0.97,
    f"R²: {r2:.2f}\nRMSE: {rmse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=13,
    verticalalignment="top",
)
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map.keys()))
cbar.set_ticklabels(list(season_map.values()))

for size in [20 + min_size_offset, 50 + min_size_offset, 100 + min_size_offset]:
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
    title="Diatom Percentage",
)
plt.show()

# Residual plot
residuals_normal = 10**y_test - 10**y_pred
residual_plot_df = pd.DataFrame(
    {
        "Predicted Diatom Concentration": 10**y_pred,
        "Residuals": residuals_normal,
    },
    index=y_test_original_index,
)
residual_plot_df = residual_plot_df.merge(
    HL_data_month[["Season", "diatom_percent"]],
    left_index=True,
    right_index=True,
    how="left",
)
desired_season_order_residual = ["Spring", "Summer", "Fall"]
residual_plot_df["Season"] = pd.Categorical(
    residual_plot_df["Season"],
    categories=desired_season_order_residual,
    ordered=True,
)
season_map_residual = {
    code: season
    for code, season in enumerate(residual_plot_df["Season"].cat.categories)
}
min_size_offset = 5
plt.figure(figsize=(5, 3))
scatter = plt.scatter(
    residual_plot_df["Predicted Diatom Concentration"],
    residual_plot_df["Residuals"],
    c=residual_plot_df["Season"].astype("category").cat.codes,
    s=residual_plot_df["diatom_percent"] + min_size_offset,
    cmap="rainbow",
    alpha=0.5,
)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Biomass (μg/L)", fontsize=12)
plt.ylabel("Residuals (μg/L)", fontsize=12)
plt.title("Diatoms")
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map_residual.keys()))
cbar.set_ticklabels(list(season_map_residual.values()))
cbar.set_label("Season")

for size in [20 + min_size_offset, 50 + min_size_offset, 100 + min_size_offset]:
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
    title="Percentage",
)
plt.show()

# SHAP feature interpretation
masker = shap.maskers.Independent(X_train, max_samples=X_train.shape[0])
explainer = shap.Explainer(
    best_model.named_steps["svr"].predict,
    masker,
)
shap_values = explainer(X_test)
feature_names = list(X_filtered.columns)

if "HPLCHLA" in feature_names:
    hplchla_index = feature_names.index("HPLCHLA")
    feature_names[hplchla_index] = "CHL"

shap_values.feature_names = feature_names
plt.rcParams["figure.figsize"] = (3, 4)
shap.plots.beeswarm(
    shap_values,
    max_display=10,
    order=shap_values.abs.mean(0),
)
