"""
SPA + SVR Feature Selection for each Algal Taxa
Author: Punya
"""

#Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Load CSV file
SAMRef_data = pd.read_csv(r'/content/Reflectance_SAM_2019-2024_cdom160.csv')
Absorption_data = pd.read_csv(r'/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv')
#Rename the coloumn
SAMRef_data.rename(columns={'isdata$SAMPLE_ID': 'SAMPLE_ID'}, inplace=True)
#Merge two dataframes
Final_df = pd.merge(SAMRef_data, Absorption_data, on='SAMPLE_ID')

import datetime
#Convert to pandas datetime format
Final_df['SAMPLE_DATE'] = pd.to_datetime(Final_df['Date'])#, format='%d-%m-%Y')
#set 3 coloumsn with year month and day
Final_df['YEAR'] = Final_df['SAMPLE_DATE'].dt.year
Final_df['MONTH'] = Final_df['SAMPLE_DATE'].dt.month
Final_df['DAY'] = Final_df['SAMPLE_DATE'].dt.day
#Filter the data to a depth of less than 20 meters
Filtered_data = Final_df[Final_df['DEPTH'] < 20]

# Print columns that might match our target taxa to find correct names
target_keywords = ['diat', 'dino', 'hapto', 'chloro', 'HPLCHLA']
matching_cols = [col for col in Filtered_data.columns if any(kw in col.lower() for kw in target_keywords)]
print("Matching columns in Filtered_data:", matching_cols)

# We will use this to dynamically match or manually adjust the names
taxa_list = []
for key in ['diatom', 'dino', 'hapto', 'chloro']:
    match = [col for col in Filtered_data.columns if key in col.lower()]
    if match:
        taxa_list.append(match[0])
print("Identified taxa columns:", taxa_list)

percent_dfs = []
for taxa in taxa_list:
    taxa_biomass = Filtered_data[taxa].div(Filtered_data['HPLCHLA'], axis=0)
    taxa_percent = (taxa_biomass * 100).rename(f'{taxa}_percent')
    percent_dfs.append(taxa_percent)

if percent_dfs:
    all_taxa_percent = pd.concat(percent_dfs, axis=1)
    Reflection_taxa = pd.concat([Filtered_data, all_taxa_percent], axis=1)
    print("Successfully created Reflection_taxa with columns:", [col for col in Reflection_taxa.columns if 'percent' in col])
else:
    print("No matching taxa columns were found.")

###Seasonal Analysis###
#Group the monthly data and take the mean, excluding non-numeric columns
HL_data_month = Reflection_taxa.set_index(Filtered_data.SAMPLE_DATE.dt.month)
#HL_data_month.head()
#set the coloumn name
HL_data_month.index.name = 'Months' # Set the name of the index
# Use the rename method on the index to change the labels
HL_data_month = HL_data_month.rename(index={1:'January', 2: 'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8: 'August', 9:'September', 10:'October',11:'November',12:'December'})
#HL_data_month.head()
# Reset the index to make 'Months' a column
HL_data_month = HL_data_month.reset_index()

#Seasonal absorption spectra
#Add a coloum indicating season of the data collected month
def get_season(month):
    if month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    elif month in ['September', 'October', 'November']:
        return 'Fall'
    else:
        return 'Other' # Or handle other months as needed

HL_data_month['Season'] = HL_data_month['Months'].apply(get_season)

# Create separate datasets for each taxon
# 1. Diatom dataset: Drop 'dino1', 'hapto6', and 'chloro' columns
diatom_dataset = Filtered_data.drop(columns=['dino1', 'hapto6', 'chloro'], errors='ignore')
# 2. Dinoflagellates dataset: Drop other taxa
dinoflagellates_dataset = Filtered_data.drop(columns=['diatom', 'hapto6', 'chloro'], errors='ignore')
# 3. Haptophytes dataset: Drop other taxa
haptophytes_dataset = Filtered_data.drop(columns=['diatom', 'dino1', 'chloro'], errors='ignore')
# 4. Chlorophytes dataset: Drop other taxa
chlorophytes_dataset = Filtered_data.drop(columns=['diatom', 'dino1', 'hapto6'], errors='ignore')

# Store in cleaned variable to keep consistency with the notebook flow
Filtered_data_diatom= diatom_dataset.dropna()
Filtered_data_dinoflagellates = dinoflagellates_dataset.dropna()
Filtered_data_haptophytes = haptophytes_dataset.dropna()
Filtered_data_chlorophytes = chlorophytes_dataset.dropna()

import numpy as np
# Map each dataset variable to its main taxonomic column name and specific (lower, upper) percentiles
taxa_datasets = {
    'diatom': ('Filtered_data_diatom', 'diatom', 2, 98),
    'dinoflagellates': ('Filtered_data_dinoflagellates', 'dino1', 2, 98),
    'haptophytes': ('Filtered_data_haptophytes', 'hapto6', 2, 98),
    'chlorophytes': ('Filtered_data_chlorophytes', 'chloro', 2, 98)
}

# Apply the specific percentile filtering to each dataset
for taxa_name, (df_name, col_name, lower_percentile, upper_percentile) in taxa_datasets.items():
    df = globals()[df_name]
    # Consider only positive values for percentile calculation
    values_for_percentile = df[col_name][df[col_name] > 0]
    if not values_for_percentile.empty:
        lower_bound = np.percentile(values_for_percentile, lower_percentile)
        upper_bound = np.percentile(values_for_percentile, upper_percentile)
        original_num_rows = df.shape[0]
        # Filter to keep values within bounds and strictly positive
        filtered_df = df[
            (df[col_name] >= lower_bound) &
            (df[col_name] <= upper_bound) &
            (df[col_name] > 0)
        ].copy()
        # Update the global dataframe variable
        globals()[df_name] = filtered_df
        print(f"{taxa_name.capitalize()}: Applied {lower_percentile}%-{upper_percentile}% filtering. Removed {original_num_rows - filtered_df.shape[0]} rows. New row count: {filtered_df.shape[0]}")
    else:
        print(f"Warning: No positive values found for {taxa_name} in {df_name}.")

## Filter-Based Band Reduction ##
# List of selected reflectance bands to extract
bands_to_keep = [
    'Refl400nm', 'Refl405nm', 'Refl410nm', 'Refl415nm', 'Refl420nm',
    'Refl425nm', 'Refl430nm', 'Refl435nm', 'Refl440nm', 'Refl445nm', 'Refl450nm', 'Refl455nm', 'Refl460nm', 'Refl465nm', 'Refl470nm',
    'Refl475nm', 'Refl480nm', 'Refl485nm', 'Refl490nm', 'Refl495nm', 'Refl500nm', 'Refl505nm', 'Refl510nm', 'Refl515nm', 'Refl520nm',
    'Refl525nm', 'Refl530nm', 'Refl535nm', 'Refl540nm', 'Refl545nm', 'Refl550nm', 'Refl555nm', 'Refl560nm', 'Refl565nm', 'Refl570nm',
    'Refl575nm', 'Refl580nm', 'Refl586nm', 'Refl615nm', 'Refl620nm', 'Refl625nm', 'Refl630nm', 'Refl635nm', 'Refl640nm', 'Refl642nm',
    'Refl645nm', 'Refl647nm', 'Refl650nm', 'Refl652nm', 'Refl655nm', 'Refl657nm', 'Refl660nm', 'Refl662nm', 'Refl665nm', 'Refl667nm',
    'Refl670nm', 'Refl672nm', 'Refl675nm', 'Refl677nm', 'Refl679nm', 'Refl682nm', 'Refl697nm', 'Refl699nm',
]
# Dictionary mapping the target datasets
taxa_datasets = {
    'diatom': 'Filtered_data_diatom',
    'dinoflagellates': 'Filtered_data_dinoflagellates',
    'haptophytes': 'Filtered_data_haptophytes',
    'chlorophytes': 'Filtered_data_chlorophytes'
}
for taxa_name, df_var in taxa_datasets.items():
    if df_var in globals():
        df = globals()[df_var]
        # Sub-select columns that exist in this DataFrame
        existing_bands = [b for b in bands_to_keep if b in df.columns]
        reduced_df = df[existing_bands].copy()
        # Create mapping to rename e.g., 'Refl400nm' to '400'
        column_mapping = {col: col.replace('Refl', '').replace('nm', '') for col in reduced_df.columns}
        reduced_df = reduced_df.rename(columns=column_mapping)
        # Save as the new reduced variable name, e.g., Filtered_data_diatom1
        reduced_var_name = f"{df_var}1"
        globals()[reduced_var_name] = reduced_df
        print(f"Successfully created {reduced_var_name} with shape: {reduced_df.shape}")
    else:
        print(f"Warning: Dataset {df_var} was not found in kernel state.")

import numpy as np
# --- Diatoms Dataset ---
X_diatom_raw = Filtered_data_diatom1.copy()
diatom_values = Filtered_data_diatom.loc[X_diatom_raw.index, 'diatom']
positive_diatom = diatom_values[diatom_values > 0]
log_diatom = np.log10(positive_diatom)
valid_diatom_idx = log_diatom[np.isfinite(log_diatom)].index
X_diatom = X_diatom_raw.loc[valid_diatom_idx].copy()
y_diatom = log_diatom.loc[valid_diatom_idx].copy()

# --- Dinoflagellates Dataset ---
X_dino_raw = Filtered_data_dinoflagellates1.copy()
dino_values = Filtered_data_dinoflagellates.loc[X_dino_raw.index, 'dino1']
positive_dino = dino_values[dino_values > 0]
log_dino = np.log10(positive_dino)
valid_dino_idx = log_dino[np.isfinite(log_dino)].index
X_dinoflagellate = X_dino_raw.loc[valid_dino_idx].copy()
y_dinoflagellate = log_dino.loc[valid_dino_idx].copy()

# --- Haptophytes Dataset ---
X_hapto_raw = Filtered_data_haptophytes1.copy()
hapto_values = Filtered_data_haptophytes.loc[X_hapto_raw.index, 'hapto6']
positive_hapto = hapto_values[hapto_values > 0]
log_hapto = np.log10(positive_hapto)
valid_hapto_idx = log_hapto[np.isfinite(log_hapto)].index
X_haptophyte = X_hapto_raw.loc[valid_hapto_idx].copy()
y_haptophyte = log_hapto.loc[valid_hapto_idx].copy()

# --- Chlorophytes Dataset ---
X_chloro_raw = Filtered_data_chlorophytes1.copy()
chloro_values = Filtered_data_chlorophytes.loc[X_chloro_raw.index, 'chloro']
positive_chloro = chloro_values[chloro_values > 0]
log_chloro = np.log10(positive_chloro)
valid_chloro_idx = log_chloro[np.isfinite(log_chloro)].index
X_chlorophyte = X_chloro_raw.loc[valid_chloro_idx].copy()
y_chlorophyte = log_chloro.loc[valid_chloro_idx].copy()

# Unified Dictionaries for convenience in loops
X_datasets = {
    'diatom': X_diatom,
    'dinoflagellate': X_dinoflagellate,
    'haptophyte': X_haptophyte,
    'chlorophyte': X_chlorophyte
}
y_datasets = {
    'diatom': y_diatom,
    'dinoflagellate': y_dinoflagellate,
    'haptophyte': y_haptophyte,
    'chlorophyte': y_chlorophyte
}
# Print shapes to confirm successful, isolated preparation
for taxon in X_datasets:
    print(f"{taxon.capitalize()} dataset successfully isolated: X {X_datasets[taxon].shape}, y {y_datasets[taxon].shape}")
# Default downstream baseline variables
X = X_diatom
y = y_diatom

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# --- Configuration for SPA ---
MAX_BANDS_SPA = 25
STEP_SPA = 1
# --- SPA Algorithm Implementation ---
def spa_selection(X_data, k, y_data=None):
    X_matrix = X_data.values
    n_samples, n_features = X_matrix.shape
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

# --- Optimization Function for SPA + SVR CV ---
def optimize_spa_bands(X, y, max_bands, step, SVR_params):
    best_r2 = -np.inf
    optimal_k = 0
    optimal_bands = []
    for k in range(5, max_bands + 1, step):
        if k > X.shape[1]:
            break
        try:
            current_bands = spa_selection(X, k, y)
        except Exception:
            continue
        if not current_bands:
            continue
        X_k = X[current_bands]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**SVR_params))
        ])
        r2_scores = cross_val_score(pipeline, X_k, y, cv=5, scoring=make_scorer(r2_score), n_jobs=-1)
        mean_r2 = np.mean(r2_scores)
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            optimal_k = k
            optimal_bands = current_bands
    return optimal_bands, optimal_k, best_r2

# Generic SVR parameters
generic_svr_params = {
    'kernel': 'rbf',
    'C': 10,
    'epsilon': 0.1,
    'gamma': 'scale'
}
selected_bands = {}
results = {}

# DIATOMS
print("=== Separately Processing Diatoms ===")
optimal_bands_diatom, optimal_k_diatom, best_cv_r2_diatom = optimize_spa_bands(
    X_diatom, y_diatom, MAX_BANDS_SPA, STEP_SPA, generic_svr_params
)
selected_bands['diatom'] = pd.Index(optimal_bands_diatom)
X_diatom_sel = X_diatom[optimal_bands_diatom]
scaler_diatom = StandardScaler()
X_diatom_scaled = scaler_diatom.fit_transform(X_diatom_sel)
X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(X_diatom_scaled, y_diatom, test_size=0.2, random_state=42)
model_diatom = SVR(**generic_svr_params).fit(X_train_dia, y_train_dia)
test_r2_diatom = r2_score(y_test_dia, model_diatom.predict(X_test_dia))
results['diatom'] = test_r2_diatom
print(f"Diatom - Best CV R2: {best_cv_r2_diatom:.4f}, Test R2: {test_r2_diatom:.4f}\n")

# DINOFLAGELLATES
print("=== Separately Processing Dinoflagellates ===")
optimal_bands_dino, optimal_k_dino, best_cv_r2_dino = optimize_spa_bands(
    X_dinoflagellate, y_dinoflagellate, MAX_BANDS_SPA, STEP_SPA, generic_svr_params
)
selected_bands['dinoflagellate'] = pd.Index(optimal_bands_dino)
X_dino_sel = X_dinoflagellate[optimal_bands_dino]
scaler_dino = StandardScaler()
X_dino_scaled = scaler_dino.fit_transform(X_dino_sel)
X_train_dino, X_test_dino, y_train_dino, y_test_dino = train_test_split(X_dino_scaled, y_dinoflagellate, test_size=0.2, random_state=42)
model_dino = SVR(**generic_svr_params).fit(X_train_dino, y_train_dino)
test_r2_dino = r2_score(y_test_dino, model_dino.predict(X_test_dino))
results['dinoflagellate'] = test_r2_dino
print(f"Dinoflagellate - Best CV R2: {best_cv_r2_dino:.4f}, Test R2: {test_r2_dino:.4f}\n")

# HAPTOPHYTES
print("=== Separately Processing Haptophytes ===")
optimal_bands_hapto, optimal_k_hapto, best_cv_r2_hapto = optimize_spa_bands(
    X_haptophyte, y_haptophyte, MAX_BANDS_SPA, STEP_SPA, generic_svr_params
)
selected_bands['haptophyte'] = pd.Index(optimal_bands_hapto)
X_hapto_sel = X_haptophyte[optimal_bands_hapto]
scaler_hapto = StandardScaler()
X_hapto_scaled = scaler_hapto.fit_transform(X_hapto_sel)
X_train_hap, X_test_hap, y_train_hap, y_test_hap = train_test_split(X_hapto_scaled, y_haptophyte, test_size=0.2, random_state=42)
model_hapto = SVR(**generic_svr_params).fit(X_train_hap, y_train_hap)
test_r2_hapto = r2_score(y_test_hap, model_hapto.predict(X_test_hap))
results['haptophyte'] = test_r2_hapto
print(f"Haptophyte - Best CV R2: {best_cv_r2_hapto:.4f}, Test R2: {test_r2_hapto:.4f}\n")

# CHLOROPHYTES
print("=== Separately Processing Chlorophytes ===")
optimal_bands_chloro, optimal_k_chloro, best_cv_r2_chloro = optimize_spa_bands(
    X_chlorophyte, y_chlorophyte, MAX_BANDS_SPA, STEP_SPA, generic_svr_params
)
selected_bands['chlorophyte'] = pd.Index(optimal_bands_chloro)
X_chloro_sel = X_chlorophyte[optimal_bands_chloro]
scaler_chloro = StandardScaler()
X_chloro_scaled = scaler_chloro.fit_transform(X_chloro_sel)
X_train_chl, X_test_chl, y_train_chl, y_test_chl = train_test_split(X_chloro_scaled, y_chlorophyte, test_size=0.2, random_state=42)
model_chloro = SVR(**generic_svr_params).fit(X_train_chl, y_train_chl)
test_r2_chloro = r2_score(y_test_chl, model_chloro.predict(X_test_chl))
results['chlorophyte'] = test_r2_chloro
print(f"Chlorophyte - Best CV R2: {best_cv_r2_chloro:.4f}, Test R2: {test_r2_chloro:.4f}\n")
selected_bands['SPA + SVR CV'] = selected_bands['diatom']

#print the selected number of bands for each algal taxa
for taxon, bands in selected_bands.items():
    if taxon != 'SPA + SVR CV':
        print(f"{taxon.capitalize()}: Selected {len(bands)} bands")

import numpy as np
import matplotlib.pyplot as plt
taxa_keys = ['diatom', 'dinoflagellate', 'haptophyte', 'chlorophyte']
taxa_labels = ['Diatoms', 'Dinoflagellates', 'Haptophytes', 'Chlorophytes']
colors = ['limegreen', 'darkorange', 'dodgerblue', 'crimson']
fig, ax = plt.subplots(figsize=(8, 5))
for i, (taxon, label, color) in enumerate(zip(taxa_keys, taxa_labels, colors)):
    wavelengths = [int(x) for x in selected_bands[taxon]]
    ymin = 0.15 + i * 0.2
    ymax = 0.30 + i * 0.2
    ax.vlines(wavelengths, ymin=ymin, ymax=ymax, color=color, linewidth=3.5, label=label)
ax.set_xlim(380, 720)
ax.set_ylim(0, 1)
ax.set_xlabel("Wavelength (nm)", fontsize=14, labelpad=15)
ax.set_xticks(np.arange(400, 701, 20))
y_tick_positions = [0.225 + i * 0.2 for i in range(len(taxa_keys))]
ax.set_yticks(y_tick_positions)
ax.set_yticklabels([])

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
    
ax.annotate("", xy=(720, 0.05), xytext=(380, 0.05), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
ax.annotate("", xy=(380, 0.95), xytext=(380, 0.05), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

for y_val in y_tick_positions:
    ax.plot([377, 380], [y_val, y_val], color='black', lw=1.5)

ax.tick_params(axis='x', length=5, width=1.5, rotation=90, labelsize=11)
ax.tick_params(axis='y', length=5, width=1.5)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
