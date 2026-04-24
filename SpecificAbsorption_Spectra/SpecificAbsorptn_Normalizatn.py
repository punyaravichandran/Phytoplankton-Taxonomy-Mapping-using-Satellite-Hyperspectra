"""
Specific Absorption Normalization
Author: Punya
"""

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
FILE_PATH = "data/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
df = pd.read_csv(FILE_PATH)

# Convert date
df['SAMPLE_DATE'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
# Filter <20 m
df = df[df['DEPTH'] < 20]

# HELPER FUNCTIONS
def get_wavelength_columns(df):
    """Extract wavelength columns (wvXXXnm)"""
    return [col for col in df.columns if col.startswith('wv')]

def rename_wavelengths(df):
    """Rename wv400nm → 400"""
    return df.rename(columns=lambda x: x.replace('wv', '').replace('nm', '') if 'wv' in x else x)

def compute_specific_absorption(absorption_df, chl):
    """Divide absorption by chlorophyll"""
    return absorption_df.div(chl, axis=0)

def compute_taxa_percent(df):
    """Compute % biomass contribution"""
    taxa = ['diatom','dino1','hapto6','dictyo','chloro']
    percent = df[taxa].div(df['HPLCHLA'], axis=0) * 100

    return percent.rename(columns={
        'diatom': 'diatom_percent',
        'dino1': 'dino_percent',
        'hapto6': 'hapto_percent',
        'dictyo': 'dictyo_percent',
        'chloro': 'chloro_percent'
    })

def dominant_taxa_mean(df, taxa_col):
    """Return mean spectra where taxa >80%"""
    subset = df[df[taxa_col] > 80].mean()
    subset.index.name = 'Wavebands'

    result = subset.reset_index()
    result = result[~result['Wavebands'].str.contains('percent')]

    return result

# ABSORPTION PROCESSING
wv_cols = get_wavelength_columns(df)

absorption = df[wv_cols]
absorption = rename_wavelengths(absorption)
specific_abs = compute_specific_absorption(absorption, df['HPLCHLA'])

# TAXA PROCESSING
taxa_percent = compute_taxa_percent(df)
combined = pd.concat([specific_abs, taxa_percent], axis=1)

# Extract dominant taxa
diatoms = dominant_taxa_mean(combined, 'diatom_percent')
dino = dominant_taxa_mean(combined, 'dino_percent')
hapto = dominant_taxa_mean(combined, 'hapto_percent')
chloro = dominant_taxa_mean(combined, 'chloro_percent')

# Rename columns
diatoms.columns = ['Wavebands', 'Diatoms']
dino.columns = ['Wavebands', 'Dinoflagellates']
hapto.columns = ['Wavebands', 'Haptophytes']
chloro.columns = ['Wavebands', 'Chlorophytes']

# MERGE TAXA
spec_all = diatoms.merge(dino, on='Wavebands', how='left') \
                  .merge(hapto, on='Wavebands', how='left') \
                  .merge(chloro, on='Wavebands', how='left')
# Mixed population
spec_all['Mixed_Population'] = spec_all.drop(columns=['Wavebands']).sum(axis=1)

# NORMALIZATION (440 nm)
spec_all['Wavebands'] = pd.to_numeric(spec_all['Wavebands'])
ref_440 = spec_all[spec_all['Wavebands'] == 440]

norm = spec_all.copy()
for col in ['Diatoms','Dinoflagellates','Haptophytes','Chlorophytes','Mixed_Population']:
    norm[col] = norm[col] / ref_440[col].values[0]

# PLOTTING
plt.figure(figsize=(8,5))
for col in ['Diatoms','Dinoflagellates','Haptophytes','Chlorophytes','Mixed_Population']:
    plt.plot(norm['Wavebands'], norm[col], label=col)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Specific Absorption (440 nm)')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()