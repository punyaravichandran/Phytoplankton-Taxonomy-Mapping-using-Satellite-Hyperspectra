# -*- coding: utf-8 -*-
"""
Diatom Biomass prediction from reflectance
Author: Punya
"""

# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, RFE
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import optuna
import shap
import joblib
from matplotlib_venn import venn2

# LOAD & PREPROCESS DATA
SAMRef = pd.read_csv('/content/Reflectance_SAM_2019-2024_cdom160.csv')
Absorption = pd.read_csv('/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv')
SAMRef.rename(columns={'isdata$SAMPLE_ID': 'SAMPLE_ID'}, inplace=True)
df = pd.merge(SAMRef, Absorption, on='SAMPLE_ID')

# Date handling
df['SAMPLE_DATE'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['YEAR'] = df['SAMPLE_DATE'].dt.year
df['MONTH'] = df['SAMPLE_DATE'].dt.month
# Depth filtering
df = df[df['DEPTH'] < 20]

# TARGET VARIABLE
df = df[df['diatom'] > 0].dropna()
df['logDiatom'] = np.log10(df['diatom'])

# FEATURE SELECTION DATA
spectral_cols = [col for col in df.columns if 'Refl' in col]
X = df[spectral_cols]
y = df['logDiatom']

# FEATURE SELECTION METHODS
def variance_selection(X):
    selector = VarianceThreshold()
    selector.fit(X)
    return X.columns[selector.get_support()]

def mutual_info_selection(X, y, k=60):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]

def correlation_selection(X, y, k=60):
    corr = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
    return corr.sort_values(ascending=False).head(k).index

def rfe_selection(X, y, k=60):
    model = SVR(kernel='linear')
    selector = RFE(model, n_features_to_select=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]

# Apply methods
bands_var = variance_selection(X)
bands_mi = mutual_info_selection(X, y)
bands_corr = correlation_selection(X, y)
bands_rfe = rfe_selection(X, y)

# BAND INTERSECTION
set_mi = set(bands_mi)
set_rfe = set(bands_rfe)
overlap_bands = list(set_mi & set_rfe)
print(f"Selected overlap bands: {len(overlap_bands)}")

# DATA PREPARATION
X_model = X[overlap_bands].copy()
# Add chlorophyll proxy
X_model['HPLCHLA'] = np.log10(df['HPLCHLA'])
# Remove extreme values
mask = y >= -8
X_model = X_model.loc[mask]
y = y.loc[mask]

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_model)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# GRID SEARCH
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
print("Grid Best R2:", grid.best_score_)

# OPTUNA OPTIMIZATION
def objective(trial):
    model = SVR(
        C=trial.suggest_float('C', 1e-2, 1e3, log=True),
        epsilon=trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
        kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        degree=trial.suggest_int('degree', 2, 5)
    )

    pipe = make_pipeline(StandardScaler(), model)

    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best Optuna R2:", study.best_value)
print("Best Params:", study.best_params)

# FINAL MODEL
best = study.best_params
model = SVR(**best)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# EVALUATION
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

# VISUALIZATION
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVR Prediction")
plt.grid(True)
plt.show()

# RESIDUALS
residuals = y_test - y_pred

plt.figure(figsize=(5,3))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# SAVE MODEL
joblib.dump(model, "svr_diatom_model.joblib")
joblib.dump(scaler, "scaler.joblib")

# SHAP INTERPRETATION
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10)

