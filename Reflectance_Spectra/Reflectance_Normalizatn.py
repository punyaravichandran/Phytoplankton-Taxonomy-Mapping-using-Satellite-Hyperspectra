"""
Simulated Reflectance Normalization
Author: Punya
"""

## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

## Load and Merge CSV Datasets
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
# Merge the two datasets using SAMPLE_ID
Final_df = pd.merge(
    SAMRef_data,
    Absorption_data,
    on='SAMPLE_ID'
)
Final_df.head()

## Calculate and Plot Mean Reflectance
# Calculate the average of all numeric columns
average_of_all_columns = SAMRef_data.mean(numeric_only=True)
# Display the results
print(average_of_all_columns)
# Set the index name
average_of_all_columns.index.name = 'Wavebands'
# Convert the Series into a DataFrame
average = average_of_all_columns.reset_index()
average.columns = ['Wavebands', 'Reflectance']
# Filter the wavelength data
finalRef = average[
    average['Wavebands'] != 'SAMPLE_ID'
].copy()
finalRef.head()

# Define the x and y variables
x = finalRef['Wavebands']
y = finalRef['Reflectance']
# Set the figure size
plt.figure(figsize=(8, 4))
# Draw the line chart
plt.plot(x, y, linestyle='-')
# Label the axes
plt.xlabel('Wavelength')
plt.ylabel('Simulated Reflectance Spectra')
# Customize the x-axis
plt.xticks(rotation=90)
plt.rcParams['xtick.labelsize'] = 7
plt.tight_layout()
plt.show()

## Convert Dates and Filter Samples by Depth
# Convert sample dates to datetime format
Final_df['SAMPLE_DATE'] = pd.to_datetime(Final_df['Date'])
# Filter samples collected at depths less than 20 metres
Filtered_data = Final_df[
    Final_df['DEPTH'] < 20
].copy()

## Select Reflectance Bands from 400 to 700 nm
# Select reflectance bands
reflectance_columns = [
    f'Refl{wavelength}nm'
    for wavelength in range(400, 701)
]
Reflection = Filtered_data[reflectance_columns].copy()
# Rename wavelength columns:
# Refl400nm -> 400, Refl401nm -> 401, ..., Refl700nm -> 700
column_mapping = {
    col: col.replace('Refl', '').replace('nm', '')
    for col in Reflection.columns
}
Reflection = Reflection.rename(columns=column_mapping)
# Transpose the reflectance DataFrame
Reflection_Coloumn = Reflection.transpose()
Reflection_Coloumn.head()

## Calculate Taxonomic Percentages
# Select taxonomic groups
taxa_columns = [
    'diatom',
    'dino1',
    'hapto6',
    'chloro'
]
# Calculate taxonomic fractions relative to HPLCHLA
taxa_biomass = Filtered_data[taxa_columns].div(
    Filtered_data['HPLCHLA'],
    axis=0
)
# Convert fractions to percentages
taxa_percent = taxa_biomass * 100
# Rename the columns
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

## Calculate Mean Spectra for Each Taxonomic Group
#Samples are selected when the corresponding taxonomic percentage is greater than 80%.
# Define taxonomic percentage columns
percentage_columns = [
    'diatom_percent',
    'dino_percent',
    'hapto_percent',
    'chloro_percent'
]

### DIATOMS ###
Diatoms = Reflection_taxa[
    Reflection_taxa['diatom_percent'] > 80
].mean(numeric_only=True)
Diatoms.index.name = 'Wavebands'
Diatoms = Diatoms.rename('Diatoms')
Diatoms1 = Diatoms.reset_index()
Diatoms1 = Diatoms1[
    ~Diatoms1['Wavebands'].isin(percentage_columns)
]

### DINOFLAGELLATES ###
Dino = Reflection_taxa[
    Reflection_taxa['dino_percent'] > 80
].mean(numeric_only=True)
Dino.index.name = 'Wavebands'
Dino = Dino.rename('Dinoflagellates')
Dino1 = Dino.reset_index()
Dino1 = Dino1[
    ~Dino1['Wavebands'].isin(percentage_columns)
]

### HAPTOPHYTES ###
Hapto = Reflection_taxa[
    Reflection_taxa['hapto_percent'] > 80
].mean(numeric_only=True)
Hapto.index.name = 'Wavebands'
Hapto = Hapto.rename('Haptophytes')
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1[
    ~Hapto1['Wavebands'].isin(percentage_columns)
]

### CHLOROPHYTES ###
Chloro = Reflection_taxa[
    Reflection_taxa['chloro_percent'] > 80
].mean(numeric_only=True)
Chloro.index.name = 'Wavebands'
Chloro = Chloro.rename('Chlorophytes')
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1[
    ~Chloro1['Wavebands'].isin(percentage_columns)
]

## Combine Taxonomic Spectra with the Natural Population
# Rename the natural population reflectance column
finalRef = finalRef.rename(
    columns={'Reflectance': 'Normal_Population'}
)
# Prepare the natural population spectra
Mixed_Spectra = finalRef[
    ['Wavebands', 'Normal_Population']
].copy()
# Merge all taxonomic spectra by wavelength
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

# Convert wavelength values to numeric
Reflection_4taxa['Wavebands'] = pd.to_numeric(
    Reflection_4taxa['Wavebands'],
    errors='coerce'
)
# Sort by wavelength
Reflection_4taxa = Reflection_4taxa.sort_values(
    'Wavebands'
).reset_index(drop=True)
Reflection_4taxa.head()

## Plot Reflectance Spectra of All Taxonomic Groups
# Define the x and y variables
x = Reflection_4taxa['Wavebands']
y = Reflection_4taxa[
    [
        'Diatoms',
        'Dinoflagellates',
        'Haptophytes',
        'Chlorophytes',
        'Normal_Population'
    ]
]
# Set figure size
plt.figure(figsize=(10, 5))
# Plot reflectance spectra
for column in y.columns:
    plt.plot(
        x,
        y[column],
        linestyle='solid',
        label=column
    )
# Customize the plot
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance Spectra of Algal Taxa')
plt.xticks(
    np.arange(400, 701, step=10),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 7
plt.legend()
plt.tight_layout()
plt.show()

## Normalize Spectra Using 440 nm
# Select the row corresponding to 440 nm
Filter_440 = Reflection_4taxa.loc[
    Reflection_4taxa['Wavebands'] == 440
].iloc[0]
Filter_440

# Define the spectra to normalize
spectra_columns = [
    'Diatoms',
    'Dinoflagellates',
    'Haptophytes',
    'Chlorophytes',
    'Normal_Population'
]
# Normalize each spectrum by its value at 440 nm
normalization_values = Filter_440[spectra_columns].replace(
    0,
    np.nan
)
y_normalized = Reflection_4taxa[spectra_columns].div(
    normalization_values,
    axis=1
)
# Define the wavelength axis
x = Reflection_4taxa['Wavebands']
# Plot normalized spectra
plt.figure(figsize=(7, 4))
for column in spectra_columns:
    plt.plot(
        x,
        y_normalized[column],
        linestyle='solid',
        label=column
    )
# Customize the plot
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Normalized Reflectance', fontsize=12)
plt.xticks(
    np.arange(400, 701, step=10),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 10
plt.legend()
plt.tight_layout()
plt.show()
