#
"""
# SPA + SVR Feature Selection and Machine Learning of Diatom Biomass Detection
Author: Punya
"""

# Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import joblib
import optuna
import shap
from sklearn.model_selection import (train_test_split,cross_val_score,RepeatedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (r2_score,mean_squared_error,make_scorer)

# Load CSV files
SAMRef_data = pd.read_csv("Reflectance_SAM_2019-2024_cdom160.csv")
Absorption_data = pd.read_csv("Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv")

# Rename SAMPLE_ID column
SAMRef_data.rename(columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},inplace=True)
# Merge the two dataframes
Final_df = pd.merge(SAMRef_data,Absorption_data,on="SAMPLE_ID")
# Convert date to pandas datetime
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"],format="%d-%m-%Y")
# Create year, month, and day columns
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day
# Filter data to depth < 20 m
Filtered_data = Final_df[Final_df["DEPTH"] < 20].copy()
# Calculate diatom percentage biomass
taxa_biomass = Filtered_data["diatom"].div(Filtered_data["HPLCHLA"],axis=0)
taxa_percent = taxa_biomass * 100
taxa_percent = taxa_percent.rename("diatom_percent")
# Add diatom percentage to dataframe
Reflection_taxa = pd.concat([Filtered_data,taxa_percent],axis=1)
# Seasonal analysis
HL_data_month = Reflection_taxa.set_index(Filtered_data.SAMPLE_DATE.dt.month)
HL_data_month.index.name = "Months"
# Rename month numbers
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
        12: "December"
    }
)
# Reset index
HL_data_month = HL_data_month.reset_index()
# Define seasons
def get_season(month):
    if month in ["March", "April", "May"]:
        return "Spring"
    elif month in ["June", "July", "August"]:
        return "Summer"
    elif month in ["September", "October", "November"]:
        return "Fall"
    else:
        return "Other"

HL_data_month["Season"] = (HL_data_month["Months"].apply(get_season))
# Remove missing values
Filtered_data_cleaned = Filtered_data.dropna().copy()
# Percentile filtering of diatom concentration
# Define lower and upper percentiles
lower_percentile = 2
upper_percentile = 98
# Consider only positive diatom concentrations
diatom_values_for_percentile = (Filtered_data_cleaned["diatom"][Filtered_data_cleaned["diatom"] > 0])

if not diatom_values_for_percentile.empty:
    # Calculate percentile bounds
    lower_bound = np.percentile(
        diatom_values_for_percentile,
        lower_percentile
    )
    upper_bound = np.percentile(
        diatom_values_for_percentile,
        upper_percentile
    )
    # Number of rows before filtering
    original_num_rows = (
        Filtered_data_cleaned.shape[0]
    )
    # Apply percentile filtering
    Filtered_data_cleaned = Filtered_data_cleaned[
        (Filtered_data_cleaned["diatom"] >= lower_bound)
        & (Filtered_data_cleaned["diatom"] <= upper_bound)
        & (Filtered_data_cleaned["diatom"] > 0)
    ].copy()
    print(
        f"Removed "
        f"{original_num_rows - Filtered_data_cleaned.shape[0]} "
        f"rows due to diatom percentile filtering."
    )
    print(
        "New number of rows:",
        Filtered_data_cleaned.shape[0]
    )
else:
    print(
        "Warning: 'diatom' column is empty or contains "
        "only non-positive values."
    )
    
# Filter-Based Band Reduction
selected_wavelengths = [
    400, 403, 405, 408, 410, 413, 415, 418, 420, 422,
    425, 427, 430, 432, 435, 437, 440, 442, 445, 447,
    450, 452, 455, 457, 460, 462, 465, 467, 470, 472,
    475, 477, 480, 482, 485, 487, 490, 492, 495, 497,
    500, 502, 505, 507, 510, 512, 515, 517, 520, 522,
    525, 527, 530, 532, 535, 537, 540, 542, 545, 547,
    550, 553, 555, 558, 560, 563, 565, 568, 570, 573,
    575, 578, 580, 583, 586, 588,
    613, 615, 618, 620, 623, 625, 627, 630, 632, 635,
    637, 640, 641, 642, 643, 645, 646, 647, 648, 650,
    651, 652, 653, 655, 656, 657, 658, 660, 661, 662,
    663, 665, 666, 667, 668, 670, 671, 672, 673, 675,
    676, 677, 678, 679, 681, 682, 683, 684, 686, 687,
    688, 689, 691, 692, 693, 694, 696, 697, 698, 699
]
# Create corresponding column names
reflection_columns = [
    f"Refl{wavelength}nm"
    for wavelength in selected_wavelengths
]
# Select bands
Filtered_data_cleaned1 = (Filtered_data_cleaned[reflection_columns].copy())
# Rename wavelength columns
column_mapping = {
    column: column.replace("Refl", "").replace("nm", "")
    for column in Filtered_data_cleaned1.columns
}
Filtered_data_cleaned1.rename(columns=column_mapping,inplace=True)
# Prepare X and y
X = Filtered_data_cleaned1.copy()
# Get diatom concentration using matching indices
diatom_values = (Filtered_data.dropna().loc[X.index, "diatom"])
# Keep non-negative values
positive_diatom_values = (diatom_values[diatom_values >= 0])
# Log10 transformation
y_candidate = np.log10(positive_diatom_values).fillna(0)
# Keep only finite values
valid_indices = y_candidate[np.isfinite(y_candidate)].index
X = X.loc[valid_indices].copy()
y = y_candidate.loc[valid_indices].copy()

# SPA Feature Selection
MAX_BANDS_SPA = 25
STEP_SPA = 1
def spa_selection(X_data, k, y_data=None):
    X_matrix = X_data.values
    n_samples, n_features = X_matrix.shape
    selected_indices = []
    available_indices = list(range(n_features))
    # Initial feature selection
    if y_data is not None:
        abs_correlations = (
            X_data.corrwith(
                y_data.loc[X_data.index]
            ).abs()
        )
        initial_idx = (
            X_data.columns.get_loc(
                abs_correlations.idxmax()
            )
        )
    else:
        initial_idx = np.argmax(
            np.linalg.norm(
                X_matrix,
                axis=0
            )
        )
    selected_indices.append(
        initial_idx
    )
    available_indices.remove(
        initial_idx
    )
  
    # Successive projection
    for _ in range(k - 1):
        if not available_indices:
            break
        A = X_matrix[
            :,
            selected_indices
        ]
        max_residual_norm = -1
        next_feature_idx = -1
        for j in available_indices:
            current_feature = X_matrix[:, j]
            try:
                beta = np.linalg.lstsq(
                    A,
                    current_feature,
                    rcond=None
                )[0]
                projected_vector = A @ beta
                residual_vector = (
                    current_feature
                    - projected_vector
                )
                residual_norm = np.linalg.norm(
                    residual_vector
                )
            except np.linalg.LinAlgError:
                residual_norm = 0.0
            if residual_norm > max_residual_norm:
                max_residual_norm = (
                    residual_norm
                )
                next_feature_idx = j
        if next_feature_idx != -1:
            selected_indices.append(
                next_feature_idx
            )
            available_indices.remove(
                next_feature_idx
            )
        else:
            break
    return X_data.columns[
        selected_indices
    ].tolist()


# SPA + SVR optimization
def optimize_spa_bands(
    X,
    y,
    max_bands,
    step,
    SVR_params
):
    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []
    for k in range(
        5,
        max_bands + 1,
        step
    ):
        if k > X.shape[1]:
            break
        print(
            f"Evaluating SPA with {k} bands..."
        )
        try:
            current_bands = spa_selection(
                X,
                k,
                y
            )
        except Exception as e:
            print(
                f"SPA selection failed for "
                f"k={k}: {e}"
            )
            continue
        if not current_bands:
            print(
                "No bands selected. Skipping."
            )
            continue
        X_k = X[
            current_bands
        ]
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "svr",
                    SVR(**SVR_params)
                )
            ]
        )
        r2_scores = cross_val_score(
            pipeline,
            X_k,
            y,
            cv=5,
            scoring=make_scorer(
                r2_score
            ),
            n_jobs=-1
        )
        mean_r2 = np.mean(
            r2_scores
        )
        print(
            f"Mean R² for {k} bands: "
            f"{mean_r2:.4f}"
        )

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = (
                current_bands
            )
    return (
        optimal_bands,
        optimal_k,
        best_r2
    )

# Generic SVR parameters
generic_svr_params = {"kernel": "rbf","C": 10,"epsilon": 0.1,"gamma": "scale"}
# Run SPA + SVR CV
optimal_bands_spa, optimal_k_spa, best_r2_spa_cv = (optimize_spa_bands(X,y,MAX_BANDS_SPA,STEP_SPA,generic_svr_params))
# Store selected bands
selected_bands = {}
importance_scores = {}
selected_bands["SPA + SVR CV"] = pd.Index(optimal_bands_spa)
# Model performance comparison
results = {}
method_name = "SPA + SVR CV"
bands = selected_bands[method_name]
if not bands.empty:
        X_sel = X[
        bands
    ]
    scaler_selection = StandardScaler()
    X_sel_scaled = (
        scaler_selection.fit_transform(
            X_sel
        )
    )
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = (
        train_test_split(
            X_sel_scaled,
            y,
            test_size=0.2,
            random_state=42
        )
    )
    model = SVR(
        **generic_svr_params
    )
    model.fit(
        X_train_sel,
        y_train_sel
    )
results[
        method_name
    ] = r2_score(
        y_test_sel,
        y_pred_sel
    )
else:
    results[
        method_name
    ] = np.nan
print(
    "\nR² Scores for SPA + SVR CV:"
)
    y_pred_sel = model.predict(
        X_test_sel
    )

for method, r2 in results.items():
    print(
        f"{method}: {r2:.4f}"
    )
    print(
        f"-> Selected "
        f"{len(selected_bands[method])} bands.\n"
    )
# Store overlap/selected bands
overlap_bands = selected_bands
print(overlap_bands)

# Plot SPA-selected wavelengths
green_wavelengths = [
    int(x)
    for x in selected_bands[
        "SPA + SVR CV"
    ]
]
spa_wavelengths = green_wavelength(fig, ax = plt.subplots(figsize=(7, 4))
# Draw selected bands
ax.vlines(spa_wavelengths,ymin=0.28,ymax=0.38,color="limegreen",linewidth=4,label="SPA + SVR CV")
# Formatting
ax.set_xlim(380,720)
ax.set_ylim(0,1)
# Labels
ax.set_xlabel("Wavelength (nm)",fontsize=14)
ax.set_ylabel("Diatoms",fontsize=14)
# Ticks
ax.set_xticks(np.arange(400,701,20))
ax.set_yticks([])
# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Make axes look like arrows
ax.annotate("",xy=(720, 0),xytext=(380, 0),arrowprops=dict(arrowstyle="->",lw=1.5))
ax.annotate("",xy=(380, 1),xytext=(380, 0),arrowprops=dict(arrowstyle="->",lw=1.5))
# Hide original spines
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
# Tick formatting
ax.tick_params(axis="x",length=0,rotation=90,labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()                                 

# Machine Learning #
# Select SPA bands
X = X[overlap_bands["SPA + SVR CV"]].copy()
# Add log-transformed HPLCHLA
X["HPLCHLA"] = (np.log10(Filtered_data_cleaned["HPLCHLA"]).fillna(0))
# Target variable
y = y
# Filter target values
filtered_indices = y[y >= -8].index
X_filtered = X.loc[filtered_indices].copy()
y_filtered = y.loc[filtered_indices]
# Create seasonal feature
def get_season(month_num):
    if month_num in [3, 4, 5]:
        return "Spring"
    elif month_num in [6, 7, 8]:
        return "Summer"
    elif month_num in [9, 10, 11]:
        return "Fall"
    else:
        return "Other"

filtered_data_cleaned_seasons = (Filtered_data_cleaned["MONTH"].apply(get_season))
season_data = (filtered_data_cleaned_seasons.loc[X_filtered.index])
# encode season
season_dummies = pd.get_dummies(season_data,prefix="Season",dtype=int)
# Add seasonal features
X_filtered = pd.concat([X_filtered,season_dummies],axis=1)
# Train/validation/test split
X_train, X_temp, y_train, y_temp = (train_test_split(X_filtered,y_filtered,test_size=0.4,random_state=42))
X_val, X_test, y_val, y_test = (train_test_split(X_temp,y_temp,test_size=0.5,random_state=42))
# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# Keep original test indices
y_test_original_index = (y_test.index)

# Optuna hyperparameter tuning
def objective(trial):
    C = trial.suggest_float("C",1e-2,1e3,log=True)
    epsilon = trial.suggest_float("epsilon",1e-3,1.0,log=True)
    gamma = trial.suggest_categorical("gamma",["scale","auto"])
    kernel = trial.suggest_categorical("kernel",["rbf","poly","sigmoid"])
    degree = (trial.suggest_int(
            "degree",
            2,
            5
        )
        if kernel == "poly"
        else 3
    )
    svr = SVR(C=C,epsilon=epsilon,gamma=gamma,kernel=kernel,degree=degree)
    pipeline = make_pipeline(StandardScaler(),svr)
    cv = RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(pipeline,X_train,y_train,cv=cv,scoring=r2_scorer,n_jobs=-1)
    return scores.mean()
    
# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=150)
# Display best Optuna parameters
print("Best trial:")
trial = study.best_trial
print(f"  R² Score: "f"{trial.value:.4f}")
print("  Params:")
for key, value in trial.params.items():
    print(
        f"    {key}: {value}"
    )

# Train final optimized SVR model
best_params = trial.params
best_model = make_pipeline(
    SVR(
        C=best_params["C"],
        epsilon=best_params["epsilon"],
        gamma=best_params["gamma"],
        kernel=best_params["kernel"],
        degree=best_params.get(
            "degree",
            3
        )
    )
)
best_model.fit(X_train,y_train)
# Predict test set
y_pred = best_model.predict(X_test)
# Save model and scaler
joblib.dump(best_model,"Diatom_SVMmodel.joblib")
joblib.dump(scaler,"DiatomSVMscaler.joblib")
# Model evaluation
r2 = r2_score(y_test,y_pred)
rmse = mean_squared_error(y_test,y_pred)

# Actual vs predicted visualization
plt.figure(figsize=(4.5, 5))
# Convert predictions back to normal scale
plot_df = pd.DataFrame(
    {
        "Actual Diatom Concentration":
            10 ** y_test,

        "SVR Model Diatom Concentration":
            10 ** y_pred
    },
    index=y_test_original_index
)
# Add season and diatom percentage
plot_df = plot_df.merge(HL_data_month[["Season","diatom_percent"]],
    left_index=True,
    right_index=True,
    how="left"
)
# Season order
desired_season_order = ["Spring","Summer","Fall"]
plot_df["Season"] = pd.Categorical(
    plot_df["Season"],
    categories=desired_season_order,
    ordered=True
)
# Map season codes
season_map = {
    code: season
    for code, season in enumerate(
        plot_df["Season"].cat.categories
    )
}
# Minimum size offset
min_size_offset = 5
scatter = plt.scatter(plot_df["Actual Diatom Concentration"],
    plot_df["SVR Model Diatom Concentration"],
    c=plot_df["Season"].astype("category").cat.codes,
    s=plot_df["diatom_percent"] + min_size_offset,
    cmap="rainbow",alpha=0.5)
# Dynamic plot limits
all_values = pd.concat([plot_df["Actual Diatom Concentration"],
        plot_df["SVR Model Diatom Concentration"]])
min_val = (all_values.min() - 0.2)
max_val = (all_values.max())
# 1:1 reference line
plt.plot([min_val, max_val],[min_val, max_val],"g--")
# Labels
plt.xlabel("Actual Concentration (microgram/L)",fontsize=12)
plt.ylabel("Model Concentration (microgram/L)",fontsize=12)
plt.title("Diatoms",fontsize=12)
plt.grid(True,linestyle="--",alpha=0.6)
plt.xlim(min_val,max_val)
plt.ylim(min_val,max_val)
# Add R² and RMSE
plt.text(0.60,0.97,f"R²: {r2:.2f}\nRMSE: {rmse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=13,verticalalignment="top")
# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map.keys()))
cbar.set_ticklabels(list(season_map.values()))
# Size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:
    plt.scatter([],[],
        c="gray",alpha=0.6,
        s=size,label=f"{size - min_size_offset}%")
plt.legend(scatterpoints=1,frameon=False,
    labelspacing=1,title="Diatom Percentage")
plt.tight_layout()
plt.show()

# Model diagnostics: Residual plot
residuals_normal = (
    10 ** y_test
    - 10 ** y_pred
)
residual_plot_df = pd.DataFrame(
    {
        "Predicted Diatom Concentration":
            10 ** y_pred,
        "Residuals":
            residuals_normal
    },
    index=y_test_original_index
)
# Add season and diatom percentage
residual_plot_df = residual_plot_df.merge(
    HL_data_month[
        [
            "Season",
            "diatom_percent"
        ]
    ],
    left_index=True,
    right_index=True,
    how="left"
)

# Season order
desired_season_order_residual = [
    "Spring",
    "Summer",
    "Fall"
]
residual_plot_df["Season"] = pd.Categorical(
    residual_plot_df["Season"],
    categories=desired_season_order_residual,
    ordered=True
)
# Season mapping

season_map_residual = {
    code: season
    for code, season in enumerate(
        residual_plot_df[
            "Season"
        ].cat.categories
    )
}


# Minimum point size

min_size_offset = 5


plt.figure(
    figsize=(5, 3)
)


scatter = plt.scatter(
    residual_plot_df[
        "Predicted Diatom Concentration"
    ],
    residual_plot_df[
        "Residuals"
    ],
    c=residual_plot_df[
        "Season"
    ].astype("category").cat.codes,
    s=residual_plot_df[
        "diatom_percent"
    ] + min_size_offset,
    cmap="rainbow",
    alpha=0.5
)


# Zero-residual reference line

plt.axhline(
    0,
    color="red",
    linestyle="--"
)


plt.xlabel(
    "Predicted Biomass (microgram/L)",
    fontsize=12
)

plt.ylabel(
    "Residuals (microgram/L)",
    fontsize=12
)
plt.title(
    "Diatoms"
)

plt.grid(
    True
)


# Colorbar

cbar = plt.colorbar(
    scatter
)

cbar.set_ticks(
    list(
        season_map_residual.keys()
    )
)

cbar.set_ticklabels(
    list(
        season_map_residual.values()
    )
)

cbar.set_label(
    "Season"
)


# Percentage legend

for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:

    plt.scatter(
        [],
        [],
        c="gray",
        alpha=0.6,
        s=size,
        label=f"{size - min_size_offset}%"
    )


plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title="Percentage"
)


plt.tight_layout()

plt.show()

# 39. SHAP analysis
# ============================================================

# Create SHAP independent masker

masker = shap.maskers.Independent(
    X_train,
    max_samples=X_train.shape[0]
)


# Create SHAP explainer

explainer = shap.Explainer(
    best_model.named_steps["svr"].predict,
    masker
)


# Calculate SHAP values

shap_values = explainer(
    X_test
)


# ============================================================
# 40. Feature names for SHAP
# ============================================================

feature_names = list(
    X_filtered.columns
)


# Rename HPLCHLA to CHL

if "HPLCHLA" in feature_names:

    hplchla_index = (
        feature_names.index(
            "HPLCHLA"
        )
    )

    feature_names[
        hplchla_index
    ] = "CHL"


# Assign feature names

shap_values.feature_names = (
    feature_names
)


# ============================================================
# 41. SHAP beeswarm plot
# ============================================================

plt.figure(
    figsize=(3, 4)
)


shap.plots.beeswarm(
    shap_values,
    max_display=10,
    order=shap_values.abs.mean(0)
)
