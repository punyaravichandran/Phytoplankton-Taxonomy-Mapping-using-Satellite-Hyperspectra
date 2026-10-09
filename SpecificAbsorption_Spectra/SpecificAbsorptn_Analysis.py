"""
Seasonal & Algal Taxa Specific Absorption Analysis
Author: Punya
"""
# Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# Load CSV file
Insitu__data = pd.read_csv( r'/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv')
Insitu__data.head()

# Convert dates to pandas datetime format
Insitu__data['SAMPLE_DATE'] = pd.to_datetime(
    Insitu__data['Date'],
    format='%Y-%m-%d',
    errors='coerce'
)

Insitu__data['YEAR'] = Insitu__data['SAMPLE_DATE'].dt.year
Insitu__data['MONTH'] = Insitu__data['SAMPLE_DATE'].dt.month
Insitu__data['DAY'] = Insitu__data['SAMPLE_DATE'].dt.day

# Filter data
Filtered_data = Insitu__data

# To filter samples to depths less than 20 m, use:
# Filtered_data = Insitu__data[Insitu__data['DEPTH'] < 20]
print(
    f"Total number of rows in the filtered data: "
    f"{Filtered_data.shape[0]}"
)

# Set index to SAMPLE_DATE
HL_data_day = Filtered_data.set_index(['SAMPLE_DATE'])

# Select absorption bands: 400-700 nm
wavelengths = np.arange(400, 701)
absorption_columns = [
    f'wv{wavelength}nm' for wavelength in wavelengths
]
Absorption = HL_data_day[absorption_columns].copy()

# Rename wavelength columns
column_mapping = {
    col: col.replace('wv', '').replace('nm', '')
    for col in Absorption.columns
}

Absorption = Absorption.rename(columns=column_mapping)

# Calculate specific absorption
Specific_AbCol = Absorption.div(
    HL_data_day['HPLCHLA'],
    axis=0
)

# Transpose data
Specific_AbRow1 = Specific_AbCol.T.copy()

Specific_AbRow1.columns = pd.to_datetime(
    HL_data_day.index,
    errors='coerce'
)

Specific_AbRow1 = Specific_AbRow1.reset_index()

Specific_AbRow1 = Specific_AbRow1.rename(
    columns={'index': 'Wavebands'}
)

# Plot seasonal specific absorption
plt.rcParams["figure.figsize"] = (8, 4)

value_vars = [
    col for col in Specific_AbRow1.columns
    if col != 'Wavebands'
]

melted_df = pd.melt(
    Specific_AbRow1,
    id_vars=['Wavebands'],
    value_vars=value_vars,
    var_name='Date',
    value_name='Specific Absorption'
)

melted_df['Date'] = pd.to_datetime(melted_df['Date'])

def assign_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return np.nan

melted_df['Season'] = melted_df['Date'].dt.month.map(
    assign_season
)
melted_df = melted_df.dropna(subset=['Season'])

sns.lineplot(
    x='Wavebands',
    y='Specific Absorption',
    data=melted_df,
    hue='Season',
    errorbar='ci',
    err_style='band'
)

plt.xlabel('Wavelength (nm)', fontsize=11)
plt.ylabel(
    'Specific absorption of \n phytoplankton population (m-1)',
    fontsize=11
)
plt.xticks(rotation=90)
plt.rcParams['xtick.labelsize'] = 10
plt.legend()
plt.tight_layout()
plt.show()

# Plot normal absorption
AbRow1 = Absorption.T.copy()
AbRow1.columns = pd.to_datetime(
    HL_data_day.index,
    errors='coerce'
)
AbRow1 = AbRow1.reset_index()
AbRow1 = AbRow1.rename(
    columns={'index': 'Wavebands'}
)
value_vars = [
    col for col in AbRow1.columns
    if col != 'Wavebands'
]
melted_df = pd.melt(
    AbRow1,
    id_vars=['Wavebands'],
    value_vars=value_vars,
    var_name='Date',
    value_name='Absorption'
)
melted_df['Date'] = pd.to_datetime(melted_df['Date'])
melted_df['Season'] = melted_df['Date'].dt.month.map(
    assign_season
)
melted_df = melted_df.dropna(subset=['Season'])
sns.lineplot(
    x='Wavebands',
    y='Absorption',
    data=melted_df,
    hue='Season',
    errorbar='ci',
    err_style='band'
)
plt.xlabel('Wavelength (nm)', fontsize=11)
plt.ylabel('Absorption of phytoplankton (m-1)', fontsize=11)
plt.xticks(rotation=90)
plt.rcParams['xtick.labelsize'] = 10
plt.legend()
plt.tight_layout()
plt.show()

# Algal taxa: calculate percentages
taxa_biomass = Filtered_data[
    ['diatom', 'dino1', 'hapto6', 'dictyo', 'chloro']
].div(
    Filtered_data['HPLCHLA'],
    axis=0
)
taxa_percent = taxa_biomass * 100
taxa_percent = taxa_percent.rename(
    columns={
        'diatom': 'diatom_percent',
        'dino1': 'dino_percent',
        'hapto6': 'hapto_percent',
        'dictyo': 'dictyo_percent',
        'chloro': 'chloro_percent'
    }
)
Specific_AbCol_reset = Specific_AbCol.reset_index(drop=True)
Specific_AbCol2 = pd.concat(
    [Specific_AbCol_reset, taxa_percent.reset_index(drop=True)],
    axis=1
)

# Diatoms: mean specific absorption for >80% dominance
Diatoms = Specific_AbCol2[
    Specific_AbCol2['diatom_percent'] > 80
].mean()
Diatoms.index.name = 'Wavebands'
Diatoms = Diatoms.rename('Diatoms')
Diatoms1 = Diatoms.reset_index()
Diatoms1 = Diatoms1.drop(
    Diatoms1[
        Diatoms1['Wavebands'].isin([
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ])
    ].index
)

# Dinoflagellates
Dino = Specific_AbCol2[
    Specific_AbCol2['dino_percent'] > 80
].mean()
Dino.index.name = 'Wavebands'
Dino = Dino.rename('Dinoflagellates')
Dino1 = Dino.reset_index()
Dino1 = Dino1.drop(
    Dino1[
        Dino1['Wavebands'].isin([
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ])
    ].index
)

# Haptophytes
Hapto = Specific_AbCol2[
    Specific_AbCol2['hapto_percent'] > 80
].mean()
Hapto.index.name = 'Wavebands'
Hapto = Hapto.rename('Haptophytes')
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1.drop(
    Hapto1[
        Hapto1['Wavebands'].isin([
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ])
    ].index
)

# Dictyophytes
Dictyo = Specific_AbCol2[
    Specific_AbCol2['dictyo_percent'] > 80
].mean()
Dictyo.index.name = 'Wavebands'
Dictyo = Dictyo.rename('Dictyophytes')
Dictyo1 = Dictyo.reset_index()
Dictyo1 = Dictyo1.drop(
    Dictyo1[
        Dictyo1['Wavebands'].isin([
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ])
    ].index
)

# Chlorophytes
Chloro = Specific_AbCol2[
    Specific_AbCol2['chloro_percent'] > 80
].mean()
Chloro.index.name = 'Wavebands'
Chloro = Chloro.rename('Chlorophytes')
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1.drop(
    Chloro1[
        Chloro1['Wavebands'].isin([
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ])
    ].index
)

# Combine taxonomic mean spectra
combined_taxa = pd.merge(
    Diatoms1, Dino1,
    on='Wavebands', how='left'
)
combined_taxa1 = pd.merge(
    combined_taxa, Hapto1,
    on='Wavebands', how='left'
)
combined_taxa2 = pd.merge(
    combined_taxa1, Chloro1,
    on='Wavebands', how='left'
)
combined_taxa2['Mixed_Population'] = (
    combined_taxa2.drop('Wavebands', axis=1).sum(axis=1)
)
Mixed_population = combined_taxa2[
    ['Wavebands', 'Mixed_Population']
]
print(Mixed_population.head())

# Combine individual taxa datasets
Diatoms = Specific_AbCol2[
    Specific_AbCol2['diatom_percent'] > 80
]
Diatoms.index.name = 'Wavebands'
Diatoms1 = Diatoms.reset_index()
Diatoms1 = Diatoms1.drop(
    columns=[
        'diatom_percent', 'dino_percent',
        'hapto_percent', 'dictyo_percent',
        'chloro_percent'
    ]
)
Dino = Specific_AbCol2[
    Specific_AbCol2['dino_percent'] > 80
]
Dino.index.name = 'Wavebands'
Dino1 = Dino.reset_index()
Dino1 = Dino1.drop(
    columns=[
        'diatom_percent', 'dino_percent',
        'hapto_percent', 'dictyo_percent',
        'chloro_percent'
    ]
)
Hapto = Specific_AbCol2[
    Specific_AbCol2['hapto_percent'] > 80
]
Hapto.index.name = 'Wavebands'
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1.drop(
    columns=[
        'diatom_percent', 'dino_percent',
        'hapto_percent', 'dictyo_percent',
        'chloro_percent'
    ]
)
Dictyo = Specific_AbCol2[
    Specific_AbCol2['dictyo_percent'] > 80
]

Dictyo.index.name = 'Wavebands'
Dictyo1 = Dictyo.reset_index()
Dictyo1 = Dictyo1.drop(
    columns=[
        'diatom_percent', 'dino_percent',
        'hapto_percent', 'dictyo_percent',
        'chloro_percent'
    ]
)
Chloro = Specific_AbCol2[
    Specific_AbCol2['chloro_percent'] > 80
]
Chloro.index.name = 'Wavebands'
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1.drop(
    columns=[
        'diatom_percent', 'dino_percent',
        'hapto_percent', 'dictyo_percent',
        'chloro_percent'
    ]
)

# Create mixed-population dataset
individual_taxa_dfs = [
    Diatoms1, Dino1, Hapto1, Chloro1
]
for df in individual_taxa_dfs:
    df.drop(columns=['Taxa'], errors='ignore', inplace=True)
combined_individual_taxa = pd.concat(
    individual_taxa_dfs,
    ignore_index=True
)
Mixed_Population = combined_individual_taxa

# Combine all taxa for plotting
taxa_dfs = [
    Diatoms1, Dino1, Hapto1, Chloro1, Mixed_Population
]
taxa_names = [
    'Diatoms',
    'Dinoflagellates',
    'Haptophytes',
    'Chlorophytes',
    'Mixed_Population'
]
for i, df in enumerate(taxa_dfs):
    df['Taxa'] = taxa_names[i]

all_taxa_df = pd.concat(
    taxa_dfs,
    ignore_index=True
)
all_taxa_melt = pd.melt(
    all_taxa_df,
    id_vars=['Wavebands', 'Taxa'],
    var_name='Measurement',
    value_name='Specific Absorption'
)
all_taxa_melt['Wavebands'] = pd.to_numeric(
    all_taxa_melt['Measurement'],
    errors='coerce'
)

# Plot specific absorption by algal taxa
plt.figure(figsize=(8, 5))
sns.lineplot(
    x='Wavebands',
    y='Specific Absorption',
    hue='Taxa',
    data=all_taxa_melt,
    errorbar='ci',
    err_style='band'
)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel(
    'Specific absorption spectra (m-1)',
    fontsize=12
)
plt.xticks(np.arange(400, 701, step=10), rotation=90)
plt.rcParams['xtick.labelsize'] = 11
plt.legend()
plt.tight_layout()
plt.show()
