"""
Specific Absorption Analysis
"""

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
FILE_PATH = "data/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
df = pd.read_csv(FILE_PATH)

# DATE HANDLING
df['SAMPLE_DATE'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['SAMPLE_DATE'])
df['MONTH'] = df['SAMPLE_DATE'].dt.month

# AUTO-DETECT WAVELENGTH COLUMNS
wv_cols = [col for col in df.columns if col.startswith("wv")]

# Rename automatically: wv400nm → 400
df.rename(columns={col: col.replace("wv", "").replace("nm", "") for col in wv_cols}, inplace=True)
wv_cols = [col.replace("wv", "").replace("nm", "") for col in wv_cols]

# SPECIFIC ABSORPTION
absorption = df[wv_cols]
specific_abs = absorption.div(df['HPLCHLA'], axis=0)

# SEASONAL LABELS
def get_season(month):
    if month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    elif month in [9,10,11]:
        return "Fall"
    else:
        return np.nan

df['Season'] = df['MONTH'].apply(get_season)
df = df.dropna(subset=['Season'])

# SEASONAL SPECTRA PLOT
spec_df = specific_abs.copy()
spec_df['Season'] = df['Season']
spec_df_melt = spec_df.melt(id_vars='Season', var_name='Wavelength', value_name='Absorption')
spec_df_melt['Wavelength'] = pd.to_numeric(spec_df_melt['Wavelength'])

plt.figure(figsize=(8,4))
sns.lineplot(
    data=spec_df_melt,
    x='Wavelength',
    y='Absorption',
    hue='Season',
    errorbar='ci'
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Specific Absorption (m⁻¹)")
plt.tight_layout()
plt.show()

# TAXA PERCENTAGES
taxa_cols = ['diatom','dino1','hapto6','dictyo','chloro']
taxa_percent = df[taxa_cols].div(df['HPLCHLA'], axis=0) * 100
taxa_percent.columns = [c + "_pct" for c in taxa_cols]

# Combine
full_df = pd.concat([specific_abs, taxa_percent], axis=1)

# FUNCTION: TAXA DOMINANCE FILTER
def get_taxa_spectrum(data, taxa_col, threshold=80):
    subset = data[data[taxa_col] > threshold]
    spectrum = subset[wv_cols].mean()
    return spectrum

taxa_map = {
    'diatom_pct': 'Diatoms',
    'dino1_pct': 'Dinoflagellates',
    'hapto6_pct': 'Haptophytes',
    'chloro_pct': 'Chlorophytes'
}

spectra = []

for col, name in taxa_map.items():
    spec = get_taxa_spectrum(full_df, col)
    df_spec = spec.reset_index()
    df_spec.columns = ['Wavelength', 'Absorption']
    df_spec['Taxa'] = name
    spectra.append(df_spec)

# Combine taxa spectra
spectra_df = pd.concat(spectra)
spectra_df['Wavelength'] = pd.to_numeric(spectra_df['Wavelength'])

# TAXA COMPARISON PLOT
plt.figure(figsize=(10,5))
sns.lineplot(
    data=spectra_df,
    x='Wavelength',
    y='Absorption',
    hue='Taxa'
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Specific Absorption")
plt.tight_layout()
plt.show()