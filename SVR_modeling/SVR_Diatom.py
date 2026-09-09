"""
SPA + SVR Feature Selection and Machine Learning
for Diatom Biomass Detection
Author: Punya
"""

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
import shap
from sklearn.model_selection import (train_test_split,cross_val_score,RepeatedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import (Pipeline,make_pipeline)
from sklearn.metrics import (r2_score,mean_squared_error,make_scorer)

# LOAD DATA
SAMRef_data = pd.read_csv("Reflectance_SAM_2019-2024_cdom160.csv")
Absorption_data = pd.read_csv("Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv")
# RENAME SAMPLE ID COLUMN
SAMRef_data.rename(columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},inplace=True)
# MERGE DATASETS
Final_df = pd.merge(SAMRef_data,Absorption_data,on="SAMPLE_ID",how="inner")
# DATE PROCESSING
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"],format="%d-%m-%Y")
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day
# FILTER DATA TO DEPTH < 20 m
Filtered_data = Final_df[Final_df["DEPTH"] < 20].copy()

# CALCULATE DIATOM PERCENTAGE BIOMASS
Filtered_data["diatom_percent"] = (Filtered_data["diatom"]/ Filtered_data["HPLCHLA"]) * 100
# DEFINE SEASON
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Other"

Filtered_data["Season"] = (Filtered_data["MONTH"].apply(get_season))
# REMOVE MISSING VALUES
Filtered_data_cleaned = (Filtered_data.dropna().copy())

# DIATOM PERCENTILE FILTERING
lower_percentile = 2
upper_percentile = 98
diatom_values_for_percentile = (Filtered_data_cleaned["diatom"][Filtered_data_cleaned["diatom"] > 0])

if not diatom_values_for_percentile.empty:
    lower_bound = np.percentile(diatom_values_for_percentile,lower_percentile)
    upper_bound = np.percentile(diatom_values_for_percentile,upper_percentile)
    original_num_rows = (Filtered_data_cleaned.shape[0])

    Filtered_data_cleaned = (Filtered_data_cleaned[
            (Filtered_data_cleaned["diatom"] >= lower_bound)
            &
            (Filtered_data_cleaned["diatom"] <= upper_bound)
            &
            (Filtered_data_cleaned["diatom"] > 0)].copy()
    )

    print(
        f"Removed "
        f"{original_num_rows - Filtered_data_cleaned.shape[0]} "
        f"rows during diatom percentile filtering."
    )

    print(
        "Remaining samples:",
        Filtered_data_cleaned.shape[0]
    )

else:

    raise ValueError("No positive diatom concentrations available.")

# SPECTRAL BAND SELECTION
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

reflection_columns = [f"Refl{wavelength}nm"
    for wavelength in selected_wavelengths]

Filtered_data_cleaned1 = (Filtered_data_cleaned[reflection_columns].copy())

# RENAME SPECTRAL COLUMNS
column_mapping = {
    column: column.replace(
        "Refl", ""
    ).replace(
        "nm", ""
    )
    for column in Filtered_data_cleaned1.columns
}

Filtered_data_cleaned1.rename(columns=column_mapping,inplace=True)

# PREPARE X AND y
X = Filtered_data_cleaned1.copy()

diatom_values = (Filtered_data_cleaned["diatom"].loc[X.index])
# Only positive values are retained
positive_diatom_values = (diatom_values[diatom_values > 0])
# Log10 transformation
y = np.log10(positive_diatom_values)
# Align X and y
common_indices = X.index.intersection(y.index)
X = X.loc[common_indices].copy()
y = y.loc[common_indices].copy()
# Remove non-finite values
valid_indices = (np.isfinite(y))
X = X.loc[valid_indices].copy()
y = y.loc[valid_indices].copy()

# SPA FEATURE SELECTION
MAX_BANDS_SPA = 25
STEP_SPA = 1

def spa_selection(X_data,k,y_data=None):
    X_matrix = X_data.values
    n_samples, n_features = (X_matrix.shape)
    selected_indices = []
    available_indices = list(range(n_features))
    # Initial feature
    if y_data is not None:
        abs_correlations = (X_data.corrwith(
                y_data.loc[X_data.index]
            ).abs()
        )

        initial_idx = (X_data.columns.get_loc(
                abs_correlations.idxmax())
        )

    else:
        initial_idx = np.argmax(np.linalg.norm(
                X_matrix,axis=0))

    selected_indices.append(initial_idx)
    available_indices.remove(initial_idx)

    # Successive projection
    for _ in range(k - 1):

        if not available_indices:
            break

        A = X_matrix[:,selected_indices]
        max_residual_norm = -1
        next_feature_idx = -1

        for j in available_indices:
            current_feature = (
                X_matrix[:, j]
            )

            try:
                beta = np.linalg.lstsq(A,current_feature,
                    rcond=None)[0]
                projected_vector = (
                    A @ beta
                )

                residual_vector = (current_feature - projected_vector)

                residual_norm = (np.linalg.norm(residual_vector))

            except np.linalg.LinAlgError:

                residual_norm = 0.0

            if (residual_norm> max_residual_norm):
                max_residual_norm = (residual_norm)

                next_feature_idx = j

        if next_feature_idx != -1:

            selected_indices.append(next_feature_idx)

            available_indices.remove(next_feature_idx)
        else:

            break

    return X_data.columns[selected_indices].tolist()

# SPA + SVR OPTIMIZATION
def optimize_spa_bands(X,y,max_bands,step,svr_params):
    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []

    for k in range(5,max_bands + 1,step):

        if k > X.shape[1]:
            break

        print(
            f"Evaluating SPA with "
            f"{k} bands..."
        )
        try:
            current_bands = (spa_selection(X,k,y))

        except Exception as e:
            print(f"SPA selection failed: {e}")
            continue

        if not current_bands:

            continue

        X_k = X[current_bands]
        pipeline = Pipeline([("scaler",StandardScaler()),("svr",SVR(**svr_params))])
        r2_scores = cross_val_score(pipeline,X_k,y,cv=5,scoring=make_scorer(r2_score),n_jobs=-1)
        mean_r2 = np.mean(r2_scores)
        print(f"Mean R² for {k} bands: "
            f"{mean_r2:.4f}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = (current_bands)
    return (optimal_bands, optimal_k, best_r2)

# GENERIC SVR PARAMETERS
generic_svr_params = {
    "kernel": "rbf",
    "C": 10,
    "epsilon": 0.1,
    "gamma": "scale"
}

# RUN SPA + SVR
optimal_bands_spa, optimal_k_spa, best_r2_spa_cv = (optimize_spa_bands(X,y,MAX_BANDS_SPA,STEP_SPA,generic_svr_params))

# STORE SELECTED BANDS
selected_bands = {}
selected_bands["SPA + SVR CV"] = pd.Index(optimal_bands_spa)

# SPA MODEL PERFORMANCE
results = {}
method_name = ("SPA + SVR CV")
bands = selected_bands[method_name]

if len(bands) > 0:
    X_sel = X[bands].copy()

    scaler_selection = (StandardScaler())
    X_sel_scaled = (scaler_selection.fit_transform(X_sel))
    X_train_sel, X_test_sel, \
    y_train_sel, y_test_sel = (train_test_split(X_sel_scaled,
            y,test_size=0.2,random_state=42))

    model_selection = SVR(
        **generic_svr_params
    )
    model_selection.fit(X_train_sel,y_train_sel)
    y_pred_sel = (model_selection.predict(X_test_sel))

    results[method_name] = r2_score(y_test_sel,y_pred_sel)
else:
    results[method_name] = np.nan
print("\nR² score for SPA + SVR:")

for method, r2 in results.items():
    print(f"{method}: {r2:.4f}")
    print(f"Selected " f"{len(selected_bands[method])} " f"bands.\n")

# STORE SELECTED BANDS
overlap_bands = selected_bands

# PLOT SPA-SELECTED WAVELENGTHS
green_wavelengths = [int(x)
    for x in selected_bands["SPA + SVR CV"]
]
spa_wavelengths = (green_wavelengths)
fig, ax = plt.subplots(figsize=(7, 4))

ax.vlines(spa_wavelengths,ymin=0.28,ymax=0.38,color="limegreen",
    linewidth=4,label="SPA + SVR CV")
ax.set_xlim(380,720)
ax.set_ylim(0,1)
ax.set_xlabel("Wavelength (nm)",fontsize=14)
ax.set_ylabel("Diatoms",fontsize=14)
ax.set_xticks(np.arange(400,701,20))
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("",xy=(720, 0),
    xytext=(380, 0),arrowprops=dict(
        arrowstyle="->",lw=1.5))

ax.annotate("",xy=(380, 1),
    xytext=(380, 0),arrowprops=dict(
        arrowstyle="->",lw=1.5))

ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x",length=0,rotation=90,labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()


# MACHINE LEARNING DATASET
X_ml = X[selected_bands["SPA + SVR CV"]].copy()
# Add log-transformed HPLC chlorophyll-a
X_ml["HPLCHLA"] = np.log10(Filtered_data_cleaned["HPLCHLA"].loc[X_ml.index])
# Target
y_ml = y.loc[X_ml.index]

# REMOVE VERY LOW LOG-DIATOM VALUES
filtered_indices = (y_ml[y_ml >= -8].index)
X_filtered = X_ml.loc[filtered_indices].copy()
y_filtered = y_ml.loc[filtered_indices].copy()

# ADD SEASONAL FEATURES
season_data = (Filtered_data_cleaned["Season"]
               .loc[X_filtered.index])
season_dummies = pd.get_dummies(season_data,prefix="Season",dtype=int)
X_filtered = pd.concat([X_filtered,season_dummies],axis=1)

# TRAIN / VALIDATION / TEST SPLIT
X_train, X_temp, \
y_train, y_temp = (train_test_split(X_filtered,
        y_filtered,test_size=0.4,random_state=42))

X_val, X_test, \
y_val, y_test = (train_test_split(X_temp,
        y_temp,test_size=0.5,random_state=42))
# STANDARDIZATION
scaler = StandardScaler()
X_train_scaled = (scaler.fit_transform(X_train))
X_val_scaled = (scaler.transform(X_val))
X_test_scaled = (scaler.transform(X_test))
# Preserve test indices
y_test_original_index = (y_test.index)
# OPTUNA HYPERPARAMETER TUNING
def objective(trial):
    C = trial.suggest_float("C",1e-2,1e3,log=True)
    epsilon = trial.suggest_float("epsilon",1e-3,1.0,log=True)
    gamma = trial.suggest_categorical("gamma",["scale","auto"])

    kernel = trial.suggest_categorical("kernel",
        ["rbf","poly","sigmoid"])

    if kernel == "poly":

        degree = trial.suggest_int("degree",2,5)
    else:
        degree = 3
    svr = SVR(C=C,epsilon=epsilon,gamma=gamma,kernel=kernel,degree=degree)
    cv = RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(svr,X_train_scaled,y_train, cv=cv,scoring=r2_scorer,n_jobs=-1)
    return scores.mean()

# RUN OPTUNA
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=150)

# DISPLAY BEST PARAMETERS
print("\nBest Optuna trial:")
trial = study.best_trial
print(f"R² Score: "f"{trial.value:.4f}")
print("Best parameters:")

for key, value in (trial.params.items()):
    print(
        f"  {key}: {value}"
    )

# TRAIN FINAL SVR MODEL
best_params = (trial.params)
best_model = SVR(
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    gamma=best_params["gamma"],
    kernel=best_params["kernel"],
    degree=best_params.get("degree",3))
best_model.fit(X_train_scaled,y_train)
# TEST PREDICTION
y_pred = (best_model.predict(
        X_test_scaled))
# SAVE MODEL AND SCALER
joblib.dump(best_model,"Diatom_SVMmodel.joblib")
joblib.dump(scaler,"DiatomSVMscaler.joblib")
# MODEL EVALUATION
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))

# ACTUAL VS PREDICTED
plot_df = pd.DataFrame({
        "Actual Diatom Concentration":10 ** y_test,
        "SVR Model Diatom Concentration":10 ** y_pred
    },
    index=y_test_original_index)
# Add season and diatom percentage
plot_df = plot_df.join(Filtered_data_cleaned[["Season","diatom_percent"]],how="left)
desired_season_order = ["Spring","Summer","Fall"]
plot_df["Season"] = pd.Categorical(plot_df["Season"],
    categories=desired_season_order,ordered=True)
season_map = {code: season
    for code, season in enumerate(desired_season_order)}
min_size_offset = 5
plt.figure(figsize=(4.5, 5))
scatter = plt.scatter(plot_df["Actual Diatom Concentration"],
    plot_df["SVR Model Diatom Concentration"],
    c=plot_df["Season"].cat.codes,
    s=(plot_df["diatom_percent"]+ min_size_offset),
    cmap="rainbow",alpha=0.5)
all_values = pd.concat([plot_df[
            "Actual Diatom Concentration"],
        plot_df["SVR Model Diatom Concentration"]])
min_val = (all_values.min() - 0.2)
max_val = (all_values.max())
plt.plot([min_val, max_val],[min_val, max_val],"g--")
plt.xlabel("Actual Concentration (microgram/L)",fontsize=12)
plt.ylabel("Model Concentration (microgram/L)",fontsize=12)
plt.title("Diatoms",fontsize=12)
plt.grid(True,linestyle="--",alpha=0.6)
plt.xlim(min_val,max_val)
plt.ylim(min_val,max_val)
plt.text(0.60,0.97, f"R²: {r2:.2f}\nRMSE: {rmse:.2f}",
    transform=plt.gca().transAxes,fontsize=13,
    verticalalignment="top")
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map.keys()))
cbar.set_ticklabels(list(season_map.values()))

# Size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:
    plt.scatter([],[],c="gray",alpha=0.6,s=size,
        label=f"{size - min_size_offset}%")
plt.legend(scatterpoints=1,frameon=False,
    labelspacing=1,title="Diatom Percentage")
plt.tight_layout()
plt.show()

# RESIDUAL DIAGNOSTICS
residuals_normal = (10 ** y_test - 10 ** y_pred)
residual_plot_df = pd.DataFrame({
        "Predicted Diatom Concentration":10 ** y_pred,
        "Residuals":residuals_normal},
    index=y_test_original_index)

# Add season and percentage
residual_plot_df = residual_plot_df.join(
    Filtered_data_cleaned[["Season","diatom_percent"]],how="left")
residual_plot_df["Season"] = pd.Categorical(residual_plot_df["Season"],
    categories=desired_season_order,ordered=True)
season_map_residual = {code: season
    for code, season in enumerate(desired_season_order)}
plt.figure(figsize=(5, 3))
scatter = plt.scatter(residual_plot_df["Predicted Diatom Concentration"],
    residual_plot_df["Residuals"],
    c=residual_plot_df["Season"].cat.codes,
    s=(residual_plot_df["diatom_percent"]
        + min_size_offset),
    cmap="rainbow",alpha=0.5)

plt.axhline(0,color="red",linestyle="--")
plt.xlabel("Predicted Biomass (microgram/L)",
    fontsize=12)
plt.ylabel("Residuals (microgram/L)",fontsize=12)
plt.title("Diatoms")
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(
        season_map_residual.keys()))
cbar.set_ticklabels(list(season_map_residual.values()))
cbar.set_label("Season")
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:
    plt.scatter([],[],
        c="gray",alpha=0.6,
        s=size,label=f"{size - min_size_offset}%")
plt.legend(scatterpoints=1,frameon=False,
    labelspacing=1,title="Percentage")
plt.tight_layout()
plt.show()

# SHAP ANALYSIS
# SHAP uses the same scaled features supplied to the SVR
masker = shap.maskers.Independent(X_train_scaled,
    max_samples=X_train_scaled.shape[0])

explainer = shap.Explainer(best_model.predict,masker)
shap_values = explainer(X_test_scaled)

# SHAP FEATURE NAMES
feature_names = list(X_filtered.columns)
if "HPLCHLA" in feature_names:
    hplchla_index = (feature_names.index("HPLCHLA"))
    feature_names[hplchla_index] = "CHL"

shap_values.feature_names = (feature_names)

# SHAP BEESWARM PLOT
plt.figure(figsize=(3, 4))
shap.plots.beeswarm(
    shap_values,
    max_display=10,
    order=shap_values.abs.mean(0)
)
plt.tight_layout()
plt.show()
