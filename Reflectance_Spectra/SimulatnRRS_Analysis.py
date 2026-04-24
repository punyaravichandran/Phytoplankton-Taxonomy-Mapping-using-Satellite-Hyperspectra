"""
Simulation reflectance of Phytoplankton Taxa & its Seasonal variability
Author: Punya
"""

# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
SAMRef_data = pd.read_csv('/content/Reflectance_SAM_2019-2024.csv')
Absorption_data = pd.read_csv('/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv')

# Fix column name
SAMRef_data.rename(columns={'isdata$SAMPLE_ID': 'SAMPLE_ID'}, inplace=True)
# Merge datasets
Final_df = pd.merge(SAMRef_data, Absorption_data, on='SAMPLE_ID')

# Date Processing
Final_df['SAMPLE_DATE'] = pd.to_datetime(Final_df['Date'], format='%d-%m-%Y')
Final_df['YEAR'] = Final_df['SAMPLE_DATE'].dt.year
Final_df['MONTH'] = Final_df['SAMPLE_DATE'].dt.month
Final_df['DAY'] = Final_df['SAMPLE_DATE'].dt.day
Final_df = Final_df.set_index('SAMPLE_DATE')

# Extract Reflectance Bands
reflection_cols = [col for col in Final_df.columns if col.startswith('Refl')]
Reflection = Final_df[reflection_cols]

# Rename columns → numeric wavelength
Reflection = Reflection.rename(columns=lambda x: x.replace('Refl', '').replace('nm', ''))

# Seasonal Reflectance Plot
Reflection_T = Reflection.transpose()
Reflection_T.index.name = 'Wavebands'
Reflection_T = Reflection_T.reset_index()

# Melt for plotting
melted_df = pd.melt(
    Reflection_T,
    id_vars=['Wavebands'],
    var_name='Date',
    value_name='Reflectance'
)
melted_df['Date'] = pd.to_datetime(melted_df['Date'])

# Assign seasons
melted_df['Season'] = pd.cut(
    melted_df['Date'].dt.month,
    bins=[2, 5, 8, 11],
    labels=['Spring', 'Summer', 'Fall']
)
melted_df = melted_df.dropna(subset=['Season'])

# Plot
plt.figure(figsize=(7, 4))
sns.lineplot(
    x='Wavebands',
    y='Reflectance',
    hue='Season',
    data=melted_df,
    errorbar='ci'
)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (sr⁻¹)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Taxa Percentage Calculation
taxa_cols = ['diatom', 'dino1', 'hapto6', 'chloro']
taxa_percent = Final_df[taxa_cols].div(Final_df['HPLCHLA'], axis=0) * 100
taxa_percent = taxa_percent.rename(columns={
    'diatom': 'diatom_percent',
    'dino1': 'dino_percent',
    'hapto6': 'hapto_percent',
    'chloro': 'chloro_percent'
})
# Combine reflectance + taxa
Reflection_taxa = pd.concat([Reflection, taxa_percent], axis=1)

# Extract Dominant Taxa Spectra (>80%)
def extract_taxa_mean(df, taxa_name):
    subset = df[df[f'{taxa_name}_percent'] > 80]
    mean_spec = subset.mean()
    mean_spec.index.name = 'Wavebands'
    mean_spec = mean_spec.rename(taxa_name.capitalize())
    
    df_out = mean_spec.reset_index()
    
    # Remove percentage rows
    df_out = df_out[~df_out['Wavebands'].str.contains('percent')]
    
    return df_out

Diatoms = extract_taxa_mean(Reflection_taxa, 'diatom')
Dino = extract_taxa_mean(Reflection_taxa, 'dino')
Hapto = extract_taxa_mean(Reflection_taxa, 'hapto')
Chloro = extract_taxa_mean(Reflection_taxa, 'chloro')

# Mixed Population Reflectance
mixed_ref = Reflection.mean().reset_index()
mixed_ref.columns = ['Wavebands', 'Mixed_Population']

# Combine All Spectra
Reflection_4taxa = pd.concat(
    [Diatoms, Dino, Hapto, Chloro, mixed_ref],
    axis=1
)
Reflection_4taxa = Reflection_4taxa.loc[:, ~Reflection_4taxa.columns.duplicated()]

# Plot Taxa Reflectance
taxa_dfs = []
for name, df in zip(
    ['Diatoms', 'Dinoflagellates', 'Haptophytes', 'Chlorophytes'],
    [Diatoms, Dino, Hapto, Chloro]
):
    temp = df.copy()
    temp['Taxa'] = name
    taxa_dfs.append(temp)

all_taxa_df = pd.concat(taxa_dfs, ignore_index=True)

# Melt
all_taxa_melt = pd.melt(
    all_taxa_df,
    id_vars=['Wavebands', 'Taxa'],
    var_name='Measurement',
    value_name='Reflectance'
)
# Convert wavelength
all_taxa_melt['Wavebands'] = all_taxa_melt['Measurement'].str.extract('(\d+)').astype(float)

# Plot
plt.figure(figsize=(10, 5))
sns.lineplot(
    x='Wavebands',
    y='Reflectance',
    hue='Taxa',
    data=all_taxa_melt,
    errorbar='ci'
)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.xticks(np.arange(400, 701, 20), rotation=90)
plt.tight_layout()
plt.show()