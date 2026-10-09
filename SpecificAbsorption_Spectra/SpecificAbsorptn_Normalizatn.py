"""
Specific Absorption Normalization
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
Insitu__data = pd.read_csv(r'/content/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv')
# Convert Date column to pandas datetime format
Insitu__data['SAMPLE_DATE'] = pd.to_datetime(
    Insitu__data['Date'],
    errors='coerce'
)
# Filter the data to a depth of less than 20 meters
Filtered_data = Insitu__data[
    Insitu__data['DEPTH'] < 20
].copy()

# Monthly mean absorption spectra
HL_data_month = (
    Filtered_data
    .groupby(Filtered_data['SAMPLE_DATE'].dt.month)
    .mean(numeric_only=True)
)
HL_data_month1 = HL_data_month.rename_axis('MONTH')

# Select absorption wavelengths from 400 to 700 nm
wavelengths = np.arange(400, 701)
absorption_columns = [
    f'wv{wavelength}nm' for wavelength in wavelengths
]
Absorption1 = HL_data_month1[absorption_columns].copy()

# Calculate monthly mean specific absorption
Specific_AbCol1 = Absorption1.div(
    HL_data_month1['HPLCHLA'],
    axis=0
)
Specific_AbRow1 = Specific_AbCol1.transpose()
Specific_AbRow1.index.name = 'Wavebands'

# Use actual month labels to avoid mismatching data and months
month_names = {
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November'
}
Specific_AbRow1.columns = [
    month_names.get(month, str(month))
    for month in Specific_AbRow1.columns
]
Specific_AbRow2 = Specific_AbRow1.reset_index()
monthly_columns = [
    col for col in Specific_AbRow2.columns
    if col != 'Wavebands'
]
Specific_AbRow2['Normal_Population'] = (
    Specific_AbRow2[monthly_columns].mean(axis=1)
)

# Daily specific absorption spectra
Absorption = Filtered_data[absorption_columns].copy()

# Rename wv400nm to 400, wv401nm to 401, etc.
column_mapping = {
    col: col.replace('wv', '').replace('nm', '')
    for col in Absorption.columns
}
Absorption = Absorption.rename(columns=column_mapping)

# Calculate specific absorption by dividing by HPLC chlorophyll-a
Specific_AbCol = Absorption.div(
    Filtered_data['HPLCHLA'],
    axis=0
)

# Transpose columns to rows
Specific_AbRow = Specific_AbCol.transpose()

# Algal taxa: percentage biomass
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

# Combine specific absorption and taxonomic percentages
Specific_AbCol2 = pd.concat(
    [
        Specific_AbCol.reset_index(drop=True),
        taxa_percent.reset_index(drop=True)
    ],
    axis=1
)

# Diatoms: samples with percentage > 80
Diatoms = Specific_AbCol2[
    Specific_AbCol2['diatom_percent'] > 80
].mean(numeric_only=True)
Diatoms.index.name = 'Wavebands'
Diatoms = Diatoms.rename('Diatoms')
Diatoms1 = Diatoms.reset_index()
Diatoms1 = Diatoms1[
    ~Diatoms1['Wavebands'].isin(
        [
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ]
    )
]

# Dinoflagellates: samples with percentage > 80
Dino = Specific_AbCol2[
    Specific_AbCol2['dino_percent'] > 80
].mean(numeric_only=True)
Dino.index.name = 'Wavebands'
Dino = Dino.rename('Dinoflagellates')
Dino1 = Dino.reset_index()
Dino1 = Dino1[
    ~Dino1['Wavebands'].isin(
        [
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ]
    )
]

# Haptophytes: samples with percentage > 80
Hapto = Specific_AbCol2[
    Specific_AbCol2['hapto_percent'] > 80
].mean(numeric_only=True)
Hapto.index.name = 'Wavebands'
Hapto = Hapto.rename('Haptophytes')
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1[
    ~Hapto1['Wavebands'].isin(
        [
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ]
    )
]

# Dictyophytes: samples with percentage > 80
Dictyo = Specific_AbCol2[
    Specific_AbCol2['dictyo_percent'] > 80
].mean(numeric_only=True)
Dictyo.index.name = 'Wavebands'
Dictyo = Dictyo.rename('Dictyophytes')
Dictyo1 = Dictyo.reset_index()
Dictyo1 = Dictyo1[
    ~Dictyo1['Wavebands'].isin(
        [
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ]
    )
]

# Chlorophytes: samples with percentage > 80
Chloro = Specific_AbCol2[
    Specific_AbCol2['chloro_percent'] > 80
].mean(numeric_only=True)
Chloro.index.name = 'Wavebands'
Chloro = Chloro.rename('Chlorophytes')
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1[
    ~Chloro1['Wavebands'].isin(
        [
            'diatom_percent', 'dino_percent',
            'hapto_percent', 'dictyo_percent',
            'chloro_percent'
        ]
    )
]

# Combine the four taxonomic groups and normal population
# Extract the monthly normal-population spectrum
Mixed_Spectra = Specific_AbRow2[
    ['Wavebands', 'Normal_Population']
].copy()

# Merge individual taxonomic spectra by wavelength
Spec_abs_4taxa = Diatoms1.merge(
    Dino1, on='Wavebands', how='outer'
)
Spec_abs_4taxa = Spec_abs_4taxa.merge(
    Hapto1, on='Wavebands', how='outer'
)
Spec_abs_4taxa = Spec_abs_4taxa.merge(
    Chloro1, on='Wavebands', how='outer'
)
Spec_abs_4taxa = Spec_abs_4taxa.merge(
    Mixed_Spectra, on='Wavebands', how='left'
)
Spec_abs_4taxa['Wavebands'] = pd.to_numeric(
    Spec_abs_4taxa['Wavebands'],
    errors='coerce'
)
Spec_abs_4taxa = Spec_abs_4taxa.sort_values(
    'Wavebands'
).reset_index(drop=True)
print(Spec_abs_4taxa.head())

# Plot specific absorption spectra of algal taxa
x = Spec_abs_4taxa['Wavebands']
taxa_columns = [
    'Diatoms',
    'Dinoflagellates',
    'Haptophytes',
    'Chlorophytes',
    'Normal_Population'
]
y = Spec_abs_4taxa[taxa_columns]
plt.rcParams["figure.figsize"] = (10, 5)
plt.plot(
    x,
    y,
    linestyle='solid',
    label=[
        'Diatoms',
        'Dinoflagellates',
        'Haptophytes',
        'Chlorophytes',
        'Normal Population'
    ]
)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Specific absorption spectra of algal taxa')
plt.xticks(
    np.arange(400, 701, step=10),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 7
plt.legend()
plt.tight_layout()
plt.show()

# Normalization with 440 nm absorption
# Select the row corresponding to 440 nm
Filter_440 = Spec_abs_4taxa.loc[
    Spec_abs_4taxa['Wavebands'] == 440,
    taxa_columns
].iloc[0]
print(Filter_440)

# Normalize each spectrum by its 440 nm value
x = Spec_abs_4taxa['Wavebands']
y = Spec_abs_4taxa[taxa_columns].divide(
    Filter_440.replace(0, np.nan),
    axis=1
)
plt.rcParams["figure.figsize"] = (7, 4)
plt.plot(
    x,
    y,
    linestyle='solid',
    label=[
        'Diatoms',
        'Dinoflagellates',
        'Haptophytes',
        'Chlorophytes',
        'Mixed Population'
    ]
)
plt.xlabel('Wavelength (nm)', fontsize=11)
plt.ylabel(
    'Specific absorption normalized to 440 nm',
    fontsize=11
)
plt.xticks(
    np.arange(400, 701, step=10),
    rotation=90
)
plt.rcParams['xtick.labelsize'] = 10
plt.legend()
plt.tight_layout()
plt.show()
