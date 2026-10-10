"""
Simulation Reflectance of Phytoplankton Taxa & Its Seasonal Variability
Author: Punya
"""

# Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

## Load and Merge the Data
# Load CSV files
SAMRef_data = pd.read_csv(
    r'/content/Reflectance_SAM_2019-2024_cdom160.csv'
)
Absorption_data = pd.read_csv(
    r'/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv'
)

# Rename the sample ID column
SAMRef_data.rename(
    columns={'isdata$SAMPLE_ID': 'SAMPLE_ID'},
    inplace=True
)

# Merge reflectance and absorption datasets
Final_df = pd.merge(
    SAMRef_data,
    Absorption_data,
    on='SAMPLE_ID'
)
Final_df.head()

## Calculate Mean Reflectance
# Calculate the mean of all numeric columns
average_of_all_columns = SAMRef_data.mean(numeric_only=True)
# Set the index name
average_of_all_columns.index.name = 'Wavebands'
# Convert the Series into a DataFrame
average = average_of_all_columns.reset_index()
average.columns = ['Wavebands', 'Reflectance']

# Exclude SAMPLE_ID if present
finalRef = average[
    average['Wavebands'] != 'SAMPLE_ID'
].copy()

# Rename the mean reflectance column
finalRef = finalRef.rename(
    columns={'Reflectance': 'Natural_Population'}
)

## Convert Dates and Filter Samples
# Convert the sample date to datetime format
Final_df['SAMPLE_DATE'] = pd.to_datetime(Final_df['Date'])
# Extract year, month, and day
Final_df['YEAR'] = Final_df['SAMPLE_DATE'].dt.year
Final_df['MONTH'] = Final_df['SAMPLE_DATE'].dt.month
Final_df['DAY'] = Final_df['SAMPLE_DATE'].dt.day

# Filter samples collected at depths less than 20 metres
Filtered_data = Final_df[
    Final_df['DEPTH'] < 20
].copy()

# Set SAMPLE_DATE as the index
Filtered_data = Filtered_data.set_index('SAMPLE_DATE')

## Select and Rename Reflectance Bands
# Select reflectance bands from 400 to 700 nm
reflectance_columns = [
    f'Refl{wavelength}nm'
    for wavelength in range(400, 701)
]
Reflection = Filtered_data[reflectance_columns].copy()
# Rename wavelength columns:
# Refl400nm -> 400, Refl401nm -> 401, etc.
column_mapping = {
    column: column.replace('Refl', '').replace('nm', '')
    for column in Reflection.columns
}
Reflection = Reflection.rename(columns=column_mapping)

# Transpose the reflectance DataFrame
Reflection_Coloumn = Reflection.transpose()
# Set the index name
Reflection_Coloumn.index.name = 'Wavebands'
# Reset the index
Reflection_Coloumn = Reflection_Coloumn.reset_index()
Reflection_Coloumn.head()

## Seasonal Reflectance Spectra with Confidence Intervals
# Set figure size
plt.rcParams["figure.figsize"] = (8, 4)
# Identify all columns except Wavebands
value_vars = [
    column
    for column in Reflection_Coloumn.columns
    if column != 'Wavebands'
]
# Convert the DataFrame to long format
melted_df = pd.melt(
    Reflection_Coloumn,
    id_vars=['Wavebands'],
    value_vars=value_vars,
    var_name='Date',
    value_name='Reflection'
)
# Convert Date to datetime
melted_df['Date'] = pd.to_datetime(melted_df['Date'])
# Assign seasons based on month
melted_df['Season'] = pd.cut(
    melted_df['Date'].dt.month,
    bins=[2, 5, 8, 11],
    labels=['Spring', 'Summer', 'Fall'],
    ordered=False,
    right=True
)
# Remove samples without an assigned season
melted_df = melted_df.dropna(subset=['Season'])

# Plot seasonal reflectance spectra
sns.lineplot(
    x='Wavebands',
    y='Reflection',
    data=melted_df,
    hue='Season',
    errorbar='ci',
    err_style='band'
)
# Customize the plot
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance spectra (sr-1)', fontsize=12)
plt.xticks(rotation=90)
plt.rcParams['xtick.labelsize'] = 12
plt.legend()
plt.tight_layout()
plt.show()

## Calculate Taxonomic Biomass Percentages
# Select phytoplankton taxonomic groups
taxa_columns = [
    'diatom',
    'dino1',
    'hapto6',
    'chloro'
]
# Calculate taxonomic biomass fractions relative to HPLCHLA
taxa_biomass = Filtered_data[taxa_columns].div(
    Filtered_data['HPLCHLA'],
    axis=0
)
# Convert fractions to percentages
taxa_percent = taxa_biomass * 100
# Rename percentage columns
taxa_percent = taxa_percent.rename(
    columns={
        'diatom': 'diatom_percent',
        'dino1': 'dino_percent',
        'hapto6': 'hapto_percent',
        'chloro': 'chloro_percent'
    }
)
# Combine reflectance and taxonomic percentages
Reflection_taxa = pd.concat(
    [Reflection, taxa_percent],
    axis=1
)
Reflection_taxa.head()

## Calculate Mean Reflectance for Individual Taxa
#Samples are selected when the corresponding taxon accounts for more than 80% of the calculated taxonomic percentage.
# Identify percentage columns
percentage_columns = [
    'diatom_percent',
    'dino_percent',
    'hapto_percent',
    'chloro_percent'
]

# Calculate mean spectra for diatoms
Diatoms = Reflection_taxa[
    Reflection_taxa['diatom_percent'] > 80
].mean(numeric_only=True)
Diatoms.index.name = 'Wavebands'
Diatoms = Diatoms.rename('Diatoms')
Diatoms1 = Diatoms.reset_index()
Diatoms1 = Diatoms1[
    ~Diatoms1['Wavebands'].isin(percentage_columns)
]

# Calculate mean spectra for dinoflagellates
Dino = Reflection_taxa[
    Reflection_taxa['dino_percent'] > 80
].mean(numeric_only=True)
Dino.index.name = 'Wavebands'
Dino = Dino.rename('Dinoflagellates')
Dino1 = Dino.reset_index()
Dino1 = Dino1[
    ~Dino1['Wavebands'].isin(percentage_columns)
]

# Calculate mean spectra for haptophytes
Hapto = Reflection_taxa[
    Reflection_taxa['hapto_percent'] > 80
].mean(numeric_only=True)
Hapto.index.name = 'Wavebands'
Hapto = Hapto.rename('Haptophytes')
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1[
    ~Hapto1['Wavebands'].isin(percentage_columns)
]

# Calculate mean spectra for chlorophytes
Chloro = Reflection_taxa[
    Reflection_taxa['chloro_percent'] > 80
].mean(numeric_only=True)
Chloro.index.name = 'Wavebands'
Chloro = Chloro.rename('Chlorophytes')
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1[
    ~Chloro1['Wavebands'].isin(percentage_columns)
]

## Combine Taxa Spectra with Natural Population Reflectance
# Extract the natural population reflectance
Mixed_Spectra = finalRef[
    ['Wavebands', 'Natural_Population']
].copy()

# Combine the mean spectra for all taxa
Reflection_4taxa = Diatoms1.merge(
    Dino1,
    on='Wavebands',
    how='outer'
)
Reflection_4taxa = Reflection_4taxa.merge(
    Hapto1,
    on='Wavebands',
    how='outer'
)
Reflection_4taxa = Reflection_4taxa.merge(
    Chloro1,
    on='Wavebands',
    how='outer'
)
Reflection_4taxa = Reflection_4taxa.merge(
    Mixed_Spectra,
    on='Wavebands',
    how='outer'
)

# Sort by wavelength
Reflection_4taxa['Wavebands'] = pd.to_numeric(
    Reflection_4taxa['Wavebands'],
    errors='coerce'
)
Reflection_4taxa = Reflection_4taxa.sort_values(
    'Wavebands'
).reset_index(drop=True)
Reflection_4taxa.head()

## Plot Reflectance Spectra for All Taxa
# Define the x and y variables
x = Reflection_4taxa['Wavebands']
y = Reflection_4taxa[
    [
        'Diatoms',
        'Dinoflagellates',
        'Haptophytes',
        'Chlorophytes',
        'Natural_Population'
    ]
]

# Set figure size
plt.figure(figsize=(7, 4))
# Plot all taxa spectra
for column in y.columns:
    plt.plot(
        x,
        y[column],
        linestyle='solid',
        label=column
    )
# Customize the plot
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance spectra (sr-1)', fontsize=12)
plt.xticks(
    np.arange(400, 701, step=10),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 11
plt.legend()
plt.tight_layout()
plt.show()

## Calculate Mean HPLCHLA for Each Taxonomic Group
# Get sample dates for each taxonomic group
diatom_dates = Reflection_taxa[
    Reflection_taxa['diatom_percent'] > 80
].index
dino_dates = Reflection_taxa[
    Reflection_taxa['dino_percent'] > 80
].index
hapto_dates = Reflection_taxa[
    Reflection_taxa['hapto_percent'] > 80
].index
chloro_dates = Reflection_taxa[
    Reflection_taxa['chloro_percent'] > 80
].index

# Calculate mean HPLCHLA for each group
mean_hplchla_diatoms = Filtered_data.loc[
    diatom_dates, 'HPLCHLA'
].mean()
mean_hplchla_dino = Filtered_data.loc[
    dino_dates, 'HPLCHLA'
].mean()
mean_hplchla_hapto = Filtered_data.loc[
    hapto_dates, 'HPLCHLA'
].mean()
mean_hplchla_chloro = Filtered_data.loc[
    chloro_dates, 'HPLCHLA'
].mean()

# Calculate mean HPLCHLA for the natural population
mean_hplchla_natural_population = Filtered_data[
    'HPLCHLA'
].mean()
# Add mean HPLCHLA values to the combined spectra DataFrame
Reflection_4taxa['Diatoms_HPLCHLA'] = mean_hplchla_diatoms
Reflection_4taxa['Dinoflagellates_HPLCHLA'] = mean_hplchla_dino
Reflection_4taxa['Haptophytes_HPLCHLA'] = mean_hplchla_hapto
Reflection_4taxa['Chlorophytes_HPLCHLA'] = mean_hplchla_chloro
Reflection_4taxa['Natural_Population_HPLCHLA'] = (
    mean_hplchla_natural_population
)
Reflection_4taxa.head()

## Prepare Taxa-Specific Reflectance Data for Confidence-Interval Plotting
# Define reflectance wavelength columns
reflectance_cols = [
    str(wavelength)
    for wavelength in range(400, 701)
]
# Diatoms
diatoms_samples_df = Reflection_taxa[
    Reflection_taxa['diatom_percent'] > 80
].copy().reset_index()
Diatoms_melted = pd.melt(
    diatoms_samples_df,
    id_vars=['SAMPLE_DATE'],
    value_vars=reflectance_cols,
    var_name='Wavebands',
    value_name='Reflectance'
)
Diatoms_melted['Wavebands'] = pd.to_numeric(
    Diatoms_melted['Wavebands']
)
Diatoms_melted['Taxa'] = 'Diatoms'
# Dinoflagellates
dino_samples_df = Reflection_taxa[
    Reflection_taxa['dino_percent'] > 80
].copy().reset_index()
Dino_melted = pd.melt(
    dino_samples_df,
    id_vars=['SAMPLE_DATE'],
    value_vars=reflectance_cols,
    var_name='Wavebands',
    value_name='Reflectance'
)
Dino_melted['Wavebands'] = pd.to_numeric(
    Dino_melted['Wavebands']
)
Dino_melted['Taxa'] = 'Dinoflagellates'
# Haptophytes
hapto_samples_df = Reflection_taxa[
    Reflection_taxa['hapto_percent'] > 80
].copy().reset_index()
Hapto_melted = pd.melt(
    hapto_samples_df,
    id_vars=['SAMPLE_DATE'],
    value_vars=reflectance_cols,
    var_name='Wavebands',
    value_name='Reflectance'
)
Hapto_melted['Wavebands'] = pd.to_numeric(
    Hapto_melted['Wavebands']
)
Hapto_melted['Taxa'] = 'Haptophytes'

# Chlorophytes
chloro_samples_df = Reflection_taxa[
    Reflection_taxa['chloro_percent'] > 80
].copy().reset_index()
Chloro_melted = pd.melt(
    chloro_samples_df,
    id_vars=['SAMPLE_DATE'],
    value_vars=reflectance_cols,
    var_name='Wavebands',
    value_name='Reflectance'
)
Chloro_melted['Wavebands'] = pd.to_numeric(
    Chloro_melted['Wavebands']
)
Chloro_melted['Taxa'] = 'Chlorophytes'

## Combine Taxa and Natural Population Data
# Prepare natural population reflectance data
natural_population_samples_df = Reflection.copy().reset_index()
Natural_Population = pd.melt(
    natural_population_samples_df,
    id_vars=['SAMPLE_DATE'],
    value_vars=reflectance_cols,
    var_name='Wavebands',
    value_name='Reflectance'
)
Natural_Population['Wavebands'] = pd.to_numeric(
    Natural_Population['Wavebands']
)
Natural_Population['Taxa'] = 'Mixed Population'
# Combine all taxonomic groups and the natural population
taxa_and_natural_dfs = [
    Diatoms_melted,
    Dino_melted,
    Hapto_melted,
    Chloro_melted,
    Natural_Population
]
plot_df = pd.concat(
    taxa_and_natural_dfs,
    ignore_index=True
)
# Ensure wavelength values are numeric
plot_df['Wavebands'] = pd.to_numeric(
    plot_df['Wavebands']
)
plot_df.head()

## Plot Taxa-Specific Reflectance Spectra with Confidence Intervals
# Set figure size
plt.figure(figsize=(7, 4))
# Plot reflectance spectra for each group
sns.lineplot(
    x='Wavebands',
    y='Reflectance',
    hue='Taxa',
    data=plot_df,
    errorbar='ci',
    err_style='band'
)
# Customize the plot
plt.xlabel('Wavelength (nm)', fontsize=11)
plt.ylabel('Reflectance spectra (sr-1)', fontsize=11)
plt.xticks(
    np.arange(400, 701, step=20),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 10
plt.legend()
plt.tight_layout()
plt.show()
