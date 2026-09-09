"""
Seasonal & Algal Taxa Specific Absorption Analysis
Author: Punya
"""
# IMPORT REQUIRED PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# LOAD IN-SITU DATA
Insitu_data = pd.read_csv(
    "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)
# DATE PROCESSING
Insitu_data["SAMPLE_DATE"] = pd.to_datetime(
    Insitu_data["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Extract year, month, and day
Insitu_data["YEAR"] = Insitu_data["SAMPLE_DATE"].dt.year
Insitu_data["MONTH"] = Insitu_data["SAMPLE_DATE"].dt.month
Insitu_data["DAY"] = Insitu_data["SAMPLE_DATE"].dt.day

# SELECT DATA FOR ANALYSIS
Filtered_data = Insitu_data.copy()
# Set SAMPLE_DATE as index
HL_data_day = Filtered_data.set_index(
    "SAMPLE_DATE"
)

# EXTRACT ABSORPTION SPECTRA: 400–700 nm
absorption_columns = [
    f"wv{wavelength}nm"
    for wavelength in range(400, 701)
]

Absorption = HL_data_day[
    absorption_columns
].copy()

# RENAME ABSORPTION COLUMNS TO WAVELENGTHS
column_mapping = {
    column: column.replace("wv", "").replace("nm", "")
    for column in Absorption.columns
}

Absorption.rename(
    columns=column_mapping,
    inplace=True
)

# Convert wavelength names to integers
Absorption.columns = Absorption.columns.astype(int)
# CALCULATE SPECIFIC ABSORPTION
# Remove samples with zero or missing HPLC chlorophyll-a
valid_data = HL_data_day[
    HL_data_day["HPLCHLA"] > 0
].copy()

Absorption_valid = Absorption.loc[
    valid_data.index
]

Specific_AbCol = Absorption_valid.div(
    valid_data["HPLCHLA"],
    axis=0
)
# TRANSPOSE SPECIFIC ABSORPTION DATA
Specific_AbRow = Specific_AbCol.transpose()
Specific_AbRow.index.name = "Wavebands"
Specific_AbRow1 = (
    Specific_AbRow
    .reset_index()
)

# SEASONAL SPECIFIC ABSORPTION
# Convert from wide to long format
melted_df = pd.melt(
    Specific_AbRow1,
    id_vars=["Wavebands"],
    var_name="Date",
    value_name="Specific Absorption"
)

# Convert date to datetime
melted_df["Date"] = pd.to_datetime(
    melted_df["Date"],
    errors="coerce"
)

# Convert wavelength to numeric
melted_df["Wavebands"] = pd.to_numeric(
    melted_df["Wavebands"]
)

# ASSIGN OBSERVATIONS TO SEASONS
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return np.nan

melted_df["Season"] = (
    melted_df["Date"]
    .dt.month
    .apply(get_season)
)

# Remove winter observations
melted_df = melted_df.dropna(
    subset=["Season"]
)

# PLOT SEASONAL SPECIFIC ABSORPTION
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=melted_df,
    x="Wavebands",
    y="Specific Absorption",
    hue="Season",
    errorbar="ci",
    err_style="band"
)
plt.xlabel(
    "Wavelength (nm)",
    fontsize=11
)
plt.ylabel(
    "Specific absorption of\nphytoplankton population (m$^{-1}$)",
    fontsize=11
)

plt.xticks(
    np.arange(400, 701, 20),
    rotation=90
)

plt.xticks(fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

# CALCULATE TAXONOMIC BIOMASS PERCENTAGE
taxonomic_columns = [
    "diatom",
    "dino1",
    "hapto6",
    "dictyo",
    "chloro"
]

taxa_biomass = (
    valid_data[taxonomic_columns]
    .div(
        valid_data["HPLCHLA"],
        axis=0
    )
)

# Convert fractional contribution to percentage
taxa_percent = taxa_biomass * 100
# RENAME TAXONOMIC PERCENTAGE COLUMNS
taxa_percent.rename(
    columns={
        "diatom": "diatom_percent",
        "dino1": "dino_percent",
        "hapto6": "hapto_percent",
        "dictyo": "dictyo_percent",
        "chloro": "chloro_percent"
    },
    inplace=True
)
# COMBINE SPECIFIC ABSORPTION AND TAXONOMIC DATA
Specific_AbCol2 = pd.concat(
    [
        Specific_AbCol,
        taxa_percent
    ],
    axis=1
)
# DEFINE TAXONOMIC PERCENTAGE COLUMNS
percentage_columns = [
    "diatom_percent",
    "dino_percent",
    "hapto_percent",
    "dictyo_percent",
    "chloro_percent"
]

# CALCULATE DOMINANT TAXA SPECTRA
def calculate_dominant_spectrum(
    dataframe,
    percentage_column,
    spectrum_name
):

    dominant_data = dataframe[
        dataframe[percentage_column] > 80
    ]

    spectrum = (
        dominant_data
        .drop(columns=percentage_columns)
        .mean(numeric_only=True)
        .rename(spectrum_name)
        .rename_axis("Wavebands")
        .reset_index()
    )

    return spectrum

# DIATOMS
Diatoms1 = calculate_dominant_spectrum(
    Specific_AbCol2,
    "diatom_percent",
    "Diatoms"
)
# DINOFLAGELLATES
Dino1 = calculate_dominant_spectrum(
    Specific_AbCol2,
    "dino_percent",
    "Dinoflagellates"
)
# HAPTOPHYTES
Hapto1 = calculate_dominant_spectrum(
    Specific_AbCol2,
    "hapto_percent",
    "Haptophytes"
)
# DICTYOPHYTES
Dictyo1 = calculate_dominant_spectrum(
    Specific_AbCol2,
    "dictyo_percent",
    "Dictyophytes"
)
# CHLOROPHYTES
Chloro1 = calculate_dominant_spectrum(
    Specific_AbCol2,
    "chloro_percent",
    "Chlorophytes"
)

# CALCULATE ACTUAL MIXED-POPULATION
#     SPECIFIC ABSORPTION
Mixed_population = (
    Specific_AbCol
    .mean()
    .rename("Mixed_Population")
    .rename_axis("Wavebands")
    .reset_index()
)

# COMBINE ALL TAXONOMIC SPECTRA
combine_data = Diatoms1.merge(
    Dino1,
    on="Wavebands",
    how="outer"
)

combine_data = combine_data.merge(
    Hapto1,
    on="Wavebands",
    how="outer"
)

combine_data = combine_data.merge(
    Dictyo1,
    on="Wavebands",
    how="outer"
)

combine_data = combine_data.merge(
    Chloro1,
    on="Wavebands",
    how="outer"
)

combine_data = combine_data.merge(
    Mixed_population,
    on="Wavebands",
    how="outer"
)

# SORT BY WAVELENGTH
combine_data["Wavebands"] = pd.to_numeric(
    combine_data["Wavebands"]
)

combine_data.sort_values(
    "Wavebands",
    inplace=True
)

combine_data.reset_index(
    drop=True,
    inplace=True
)


# PLOT SPECIFIC ABSORPTION SPECTRA
x = combine_data["Wavebands"]
plot_columns = [
    "Diatoms",
    "Dinoflagellates",
    "Haptophytes",
    "Chlorophytes",
    "Mixed_Population"
]

y = combine_data[plot_columns]
plt.figure(figsize=(10, 5))
plt.plot(
    x,
    y,
    linestyle="solid"
)
plt.xlabel(
    "Wavelength (nm)"
)

plt.ylabel(
    "Specific Absorption Spectra of Algal Taxa (m$^{-1}$)"
)

plt.xticks(
    np.arange(400, 701, 20),
    rotation=90
)

plt.xticks(fontsize=9)
plt.legend(
    [
        "Diatoms",
        "Dinoflagellates",
        "Haptophytes",
        "Chlorophytes",
        "Mixed Population"
    ]
)
plt.tight_layout()
plt.show()
