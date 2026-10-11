"""
SPA feature selection and SVR for Haptophytes biomass estimation
Author: Punya
"""
# Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# Load CSV files
SAMRef_data = pd.read_csv(
    r'/content/Reflectance_SAM_2019-2024_cdom160.csv'
)
Absorption_data = pd.read_csv(
    r'/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv'
)
# Rename the column
SAMRef_data.rename(
    columns={'isdata$SAMPLE_ID': 'SAMPLE_ID'},
    inplace=True
)
# Merge the two dataframes
Final_df = pd.merge(
    SAMRef_data,
    Absorption_data,
    on='SAMPLE_ID'
)

# Convert to pandas datetime format
# Final_df['SAMPLE_DATE'] = pd.to_datetime(
#     Final_df[['YEAR', 'MONTH', 'DAY']]
# )
Final_df['SAMPLE_DATE'] = pd.to_datetime(Final_df['Date'])
# Extract year, month, and day
Final_df['YEAR'] = Final_df['SAMPLE_DATE'].dt.year
Final_df['MONTH'] = Final_df['SAMPLE_DATE'].dt.month
Final_df['DAY'] = Final_df['SAMPLE_DATE'].dt.day

# Filter the data to a depth of less than 20 meters
Filtered_data = Final_df[Final_df['DEPTH'] < 20]
# Calculate Haptophyte biomass relative to HPLC chlorophyll-a
taxa_biomass = Filtered_data['hapto6'].div(
    Filtered_data['HPLCHLA'],
    axis=0
)

# Multiply by 100 to obtain percentage biomass
taxa_percent = taxa_biomass * 100
# Rename the series
taxa_percent = taxa_percent.rename('hapto_percent')
# Add Haptophyte percentage to the dataframe
Reflection_taxa = pd.concat(
    [Filtered_data, taxa_percent],
    axis=1
)

# Set the index using the sample month
HL_data_month = Reflection_taxa.set_index(
    Filtered_data.SAMPLE_DATE.dt.month
)

# Set the index name
HL_data_month.index.name = 'Months'

# Rename month numbers to month names
HL_data_month = HL_data_month.rename(
    index={
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }
)

# Reset the index
HL_data_month = HL_data_month.reset_index()

# Assign seasons based on month names
def get_season(month):
    if month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    elif month in ['September', 'October', 'November']:
        return 'Fall'
    else:
        return 'Other'

HL_data_month['Season'] = HL_data_month['Months'].apply(get_season)
# Remove rows containing missing values
Filtered_data_cleaned = Filtered_data.dropna()

# Define lower and upper percentiles
lower_percentile = 2
upper_percentile = 98

# Select positive Haptophyte concentrations
diatom_values_for_percentile = Filtered_data_cleaned[
    Filtered_data_cleaned['hapto6'] > 0
]['hapto6']

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

    # Filter observations within the percentile bounds
    original_num_rows = Filtered_data_cleaned.shape[0]

    Filtered_data_cleaned = Filtered_data_cleaned[
        (Filtered_data_cleaned['hapto6'] >= lower_bound) &
        (Filtered_data_cleaned['hapto6'] <= upper_bound) &
        (Filtered_data_cleaned['hapto6'] > 0)
    ].copy()

    print(
        f"Removed {original_num_rows - Filtered_data_cleaned.shape[0]} "
        "rows due to Haptophyte percentile filtering."
    )

    print(
        "New number of rows in Filtered_data_cleaned:",
        Filtered_data_cleaned.shape[0]
    )

else:
    print(
        "Warning: The hapto6 column is empty or contains only "
        "non-positive values. No percentile filtering applied."
    )

# Select spectral reflectance bands
Filtered_data_cleaned1 = Filtered_data_cleaned[
    [
        'Refl400nm', 'Refl405nm', 'Refl410nm', 'Refl415nm',
        'Refl420nm', 'Refl425nm', 'Refl430nm', 'Refl435nm',
        'Refl440nm', 'Refl445nm', 'Refl450nm', 'Refl455nm',
        'Refl460nm', 'Refl465nm', 'Refl470nm', 'Refl475nm',
        'Refl480nm', 'Refl485nm', 'Refl490nm', 'Refl495nm',
        'Refl500nm', 'Refl505nm', 'Refl510nm', 'Refl515nm',
        'Refl520nm', 'Refl525nm', 'Refl530nm', 'Refl535nm',
        'Refl540nm', 'Refl545nm', 'Refl550nm', 'Refl555nm',
        'Refl560nm', 'Refl565nm', 'Refl570nm', 'Refl575nm',
        'Refl580nm', 'Refl586nm', 'Refl615nm', 'Refl620nm',
        'Refl625nm', 'Refl630nm', 'Refl635nm', 'Refl640nm',
        'Refl642nm', 'Refl645nm', 'Refl647nm', 'Refl650nm',
        'Refl652nm', 'Refl655nm', 'Refl657nm', 'Refl660nm',
        'Refl662nm', 'Refl665nm', 'Refl667nm', 'Refl670nm',
        'Refl672nm', 'Refl675nm', 'Refl677nm', 'Refl679nm',
        'Refl682nm', 'Refl697nm', 'Refl699nm'
    ]
]

# Rename spectral bands by removing 'Refl' and 'nm'
column_mapping = {
    col: col.replace('Refl', '').replace('nm', '')
    for col in Filtered_data_cleaned1.columns
}

Filtered_data_cleaned1 = Filtered_data_cleaned1.rename(
    columns=column_mapping
)

# Define X and y
X = Filtered_data_cleaned1.copy()

# Extract Haptophyte concentrations and align indices
diatom_values = Filtered_data.dropna().loc[
    X.index,
    'hapto6'
]

# Retain non-negative concentrations
positive_diatom_values = diatom_values[
    diatom_values >= 0
]

# Apply log10 transformation
y_candidate = np.log10(positive_diatom_values).fillna(0)

# Retain finite target values
valid_indices = y_candidate[
    np.isfinite(y_candidate)
].index

# Align predictor variables and target
X = X.loc[valid_indices].copy()
y = y_candidate.loc[valid_indices].copy()

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Prepare data for SPA
X_selection = Filtered_data_cleaned1.copy()
y_selection = y.copy()

# SPA configuration
MAX_BANDS_SPA = 25
STEP_SPA = 1


def spa_selection(X_data, k, y_data=None):
    """
    Perform Successive Projections Algorithm (SPA) to select k features.

    Parameters
    ----------
    X_data : pandas.DataFrame
        Spectral predictor variables.
    k : int
        Number of features to select.
    y_data : pandas.Series, optional
        Target variable used to select the initial feature.

    Returns
    -------
    list
        Selected feature names.
    """

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

    # Iteratively select features using residual norms
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
                    rcond=None
                )[0]

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

    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []

    # Evaluate different numbers of spectral bands
    for k in range(5, max_bands + 1, step):

        if k > X.shape[1]:
            break

        print(f"  Evaluating SPA with {k} bands...")

        try:
            current_bands = spa_selection(X, k, y)

        except Exception as e:
            print(f"    SPA selection failed for k={k}: {e}")
            continue

        if not current_bands:
            print(f"    No bands selected for k={k}, skipping.")
            continue

        X_k = X[current_bands]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**SVR_params))
        ])

        # Evaluate model performance using five-fold CV
        r2_scores = cross_val_score(
            pipeline,
            X_k,
            y,
            cv=5,
            scoring=make_scorer(r2_score),
            n_jobs=-1
        )

        mean_r2 = np.mean(r2_scores)

        print(f"    Mean R² for {k} bands: {mean_r2:.4f}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = current_bands

    return optimal_bands, optimal_k, best_r2


# Generic SVR parameters for band optimization
generic_svr_params = {
    'kernel': 'rbf',
    'C': 10,
    'epsilon': 0.1,
    'gamma': 'scale'
}

print("Starting SPA + SVR Cross-Validation for optimal band identification...")

optimal_bands_spa, optimal_k_spa, best_r2_spa_cv = optimize_spa_bands(
    X_selection,
    y_selection,
    MAX_BANDS_SPA,
    STEP_SPA,
    generic_svr_params
)

print(
    f"\nOptimal number of bands selected by SPA + SVR CV: {optimal_k_spa}"
)

print(
    f"Best R² achieved during optimization: {best_r2_spa_cv:.4f}"
)

print(f"Optimal bands: {optimal_bands_spa}")

# Store the selected bands
selected_bands = {}
importance_scores = {}

selected_bands['SPA + SVR CV'] = pd.Index(optimal_bands_spa)

# Evaluate the SPA + SVR CV method
results = {}

method_name = 'SPA + SVR CV'
bands = selected_bands[method_name]

if not bands.empty:

    X_sel = X_selection[bands]

    # Scale selected features
    scaler = StandardScaler()
    X_sel_scaled = scaler.fit_transform(X_sel)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel_scaled,
        y_selection,
        test_size=0.2,
        random_state=42
    )

    # Train the evaluation model
    model = SVR(**generic_svr_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[method_name] = r2_score(y_test, y_pred)

else:
    results[method_name] = np.nan

print("\nR² Scores for SPA + SVR Feature Selection:")

for method, r2 in results.items():
    print(f"{method}: {r2:.4f}")
    print(f"-> Selected {len(selected_bands[method])} bands.\n")

# Retain selected bands for machine learning
overlap_bands = selected_bands

print(overlap_bands)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Prepare features using the SPA-selected bands
X = X[overlap_bands['SPA + SVR CV']].copy()

# Add HPLC chlorophyll-a as a log-transformed predictor
X['HPLCHLA'] = np.log10(
    Filtered_data_cleaned['HPLCHLA']
).fillna(0)

# Retain target values greater than or equal to -8
filtered_indices = y[y >= -8].index

X_filtered = X.loc[filtered_indices].copy()
y_filtered = y.loc[filtered_indices]


# Assign seasons based on month numbers
def get_season(month_num):

    if month_num in [3, 4, 5]:
        return 'Spring'

    elif month_num in [6, 7, 8]:
        return 'Summer'

    elif month_num in [9, 10, 11]:
        return 'Fall'

    else:
        return 'Other'


# Create seasonal labels
filtered_data_cleaned_seasons = (
    Filtered_data_cleaned['MONTH'].apply(get_season)
)

# Align seasonal labels with X_filtered
season_data = filtered_data_cleaned_seasons.loc[
    X_filtered.index
]

# One-hot encode seasons
season_dummies = pd.get_dummies(
    season_data,
    prefix='Season',
    dtype=int
)

# Add seasonal predictors
X_filtered = pd.concat(
    [X_filtered, season_dummies],
    axis=1
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered,
    y_filtered,
    test_size=0.2,
    random_state=42
)

# Standardize the predictor variables
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Preserve original test indices for plotting
y_test_original_index = y_test.index

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of testing samples: {X_test.shape[0]}")

import optuna

from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, make_scorer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# Define the Optuna objective function
def objective(trial):

    # Suggest hyperparameter values
    C = trial.suggest_float(
        'C',
        1e-2,
        1e3,
        log=True
    )

    epsilon = trial.suggest_float(
        'epsilon',
        1e-3,
        1.0,
        log=True
    )

    gamma = trial.suggest_categorical(
        'gamma',
        ['scale', 'auto']
    )

    kernel = trial.suggest_categorical(
        'kernel',
        ['rbf', 'poly', 'sigmoid']
    )

    # Tune degree for the polynomial kernel
    degree = (
        trial.suggest_int('degree', 2, 5)
        if kernel == 'poly'
        else 3
    )

    # Define SVR model
    svr = SVR(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel,
        degree=degree
    )

    # Standardization and SVR pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        svr
    )

    # Repeated five-fold cross-validation
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    r2_scorer = make_scorer(r2_score)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=r2_scorer,
        n_jobs=-1
    )

    return scores.mean()


# Create and run Optuna study
study = optuna.create_study(
    direction='maximize'
)

study.optimize(
    objective,
    n_trials=150
)

# Display the best trial
print("Best trial:")

trial = study.best_trial

print(f"R² Score: {trial.value:.4f}")
print("Best parameters:")

for key, value in trial.params.items():
    print(f"  {key}: {value}")

import joblib

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


# Retrieve the best hyperparameters
best_params = study.best_trial.params

# Create the final SVR model
best_model = make_pipeline(
    # StandardScaler(),
    SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        degree=best_params.get('degree', 3)
    )
)

# Train the model
best_model.fit(X_train, y_train)

# Predict test data
y_pred = best_model.predict(X_test)

# Save the model and scaler
joblib.dump(
    best_model,
    "Haptophyte_SVMmodel.joblib"
)

joblib.dump(
    scaler,
    "Haptophyte_SVMscaler.joblib"
)

from sklearn.metrics import r2_score, root_mean_squared_error

# Use all test observations
filtered_y_test = y_test
filtered_y_pred = y_pred

# Calculate performance metrics
print(
    "R²:",
    r2_score(filtered_y_test, filtered_y_pred)
)

print(
    "RMSE:",
    root_mean_squared_error(filtered_y_test, filtered_y_pred)
)

# Set figure size
plt.rcParams["figure.figsize"] = (4.5, 5)

# Prepare plotting dataframe on the original concentration scale
plot_df = pd.DataFrame(
    {
        'Actual Hapto Concentration': 10**y_test,
        'SVR Model Hapto Concentration': 10**y_pred
    },
    index=y_test_original_index
)

# Merge seasonal and percentage information
plot_df = plot_df.merge(
    HL_data_month[['Season', 'hapto_percent']],
    left_index=True,
    right_index=True,
    how='left'
)

# Define seasonal order
desired_season_order = ['Spring', 'Summer', 'Fall']

plot_df['Season'] = pd.Categorical(
    plot_df['Season'],
    categories=desired_season_order,
    ordered=True
)

# Map seasonal codes to season names
season_map = {
    code: season
    for code, season in enumerate(
        plot_df['Season'].cat.categories
    )
}

# Set minimum point-size offset
min_size_offset = 5

# Create scatter plot
scatter = plt.scatter(
    plot_df['Actual Hapto Concentration'],
    plot_df['SVR Model Hapto Concentration'],
    c=plot_df['Season'].astype('category').cat.codes,
    s=plot_df['hapto_percent'] + min_size_offset,
    cmap='rainbow',
    alpha=0.5
)

# Calculate plot limits
all_values = pd.concat(
    [
        plot_df['Actual Hapto Concentration'],
        plot_df['SVR Model Hapto Concentration']
    ]
)

min_val = all_values.min() - 0.1
max_val = all_values.max()

# Add the 1:1 reference line
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'g--'
)

plt.xlabel('Actual Concentration (μg/L)', fontsize=12)
plt.ylabel('Model Concentration (μg/L)', fontsize=12)
plt.title('Haptophytes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Calculate metrics on the log-transformed target
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

plt.text(
    0.60,
    0.97,
    f'R²: {r2:.2f}\nRMSE: {rmse:.2f}',
    transform=plt.gca().transAxes,
    fontsize=13,
    verticalalignment='top'
)

# Add seasonal colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map.keys()))
cbar.set_ticklabels(list(season_map.values()))

# Add percentage size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:
    plt.scatter(
        [],
        [],
        c='gray',
        alpha=0.6,
        s=size,
        label=f'{size - min_size_offset}%'
    )

plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title='Hapto Percentage'
)

plt.show()

# Calculate residuals on the original concentration scale
residuals_normal = 10**y_test - 10**y_pred

# Create residual plot dataframe
residual_plot_df = pd.DataFrame(
    {
        'Predicted Hapto Concentration': 10**y_pred,
        'Residuals': residuals_normal
    },
    index=y_test_original_index
)

# Merge seasonal and percentage information
residual_plot_df = residual_plot_df.merge(
    HL_data_month[['Season', 'hapto_percent']],
    left_index=True,
    right_index=True,
    how='left'
)

# Define seasonal order
desired_season_order_residual = [
    'Spring',
    'Summer',
    'Fall'
]

residual_plot_df['Season'] = pd.Categorical(
    residual_plot_df['Season'],
    categories=desired_season_order_residual,
    ordered=True
)

# Map seasonal codes to season names
season_map_residual = {
    code: season
    for code, season in enumerate(
        residual_plot_df['Season'].cat.categories
    )
}

# Set minimum point-size offset
min_size_offset = 5

plt.figure(figsize=(5, 3))

scatter = plt.scatter(
    residual_plot_df['Predicted Hapto Concentration'],
    residual_plot_df['Residuals'],
    c=residual_plot_df['Season'].astype('category').cat.codes,
    s=residual_plot_df['hapto_percent'] + min_size_offset,
    cmap='rainbow',
    alpha=0.5
)

# Add zero-residual reference line
plt.axhline(
    0,
    color='red',
    linestyle='--'
)

plt.xlabel('Predicted Biomass (μg/L)', fontsize=12)
plt.ylabel('Residuals (μg/L)', fontsize=12)
plt.title('Haptophytes')
plt.grid(True)

# Add seasonal colorbar
cbar = plt.colorbar(scatter)
cbar.set_ticks(list(season_map_residual.keys()))
cbar.set_ticklabels(list(season_map_residual.values()))
cbar.set_label('Season')

# Add percentage size legend
for size in [
    20 + min_size_offset,
    50 + min_size_offset,
    100 + min_size_offset
]:
    plt.scatter(
        [],
        [],
        c='gray',
        alpha=0.6,
        s=size,
        label=f'{size - min_size_offset}%'
    )

plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title='Percentage'
)

plt.show()

import shap

# Create an independent masker
masker = shap.maskers.Independent(
    X_train,
    max_samples=X_train.shape[0]
)

# Create SHAP explainer
explainer = shap.Explainer(
    best_model.named_steps['svr'].predict,
    masker
)

# Calculate SHAP values for test data
shap_values = explainer(X_test)

# Get feature names
feature_names = list(X_filtered.columns)

# Rename HPLCHLA to CHL for display
if 'HPLCHLA' in feature_names:
    hplchla_index = feature_names.index('HPLCHLA')
    feature_names[hplchla_index] = 'CHL'

# Assign feature names
shap_values.feature_names = feature_names

# Set figure size
plt.rcParams["figure.figsize"] = (3, 4)

# Plot SHAP beeswarm
shap.plots.beeswarm(
    shap_values,
    max_display=10,
    order=shap_values.abs.mean(0)
)
