"""
Simulation Reflectance of Phytoplankton Taxa & Its Seasonal Variability
Author: Punya
"""

# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# LOAD DATA
SAMRef_data = pd.read_csv(
    "Reflectance_SAM_2019-2024_cdom160.csv"
)
# Pigment, taxonomy, and absorption data
Absorption_data = pd.read_csv(
    "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)
# RENAME SAMPLE ID COLUMN
SAMRef_data.rename(
    columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},
    inplace=True
)

# MERGE REFLECTANCE AND ABSORPTION DATA
Final_df = pd.merge(
    SAMRef_data,
    Absorption_data,
    on="SAMPLE_ID",
    how="inner"
)

# DATE PROCESSING
Final_df["SAMPLE_DATE"] = pd.to_datetime(
    Final_df["Date"],
    format="%d-%m-%Y"
)

# Extract year, month, and day
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day

# FILTER DATA BY DEPTH
Filtered_data = Final_df[Final_df["DEPTH"] < 20].copy()

# Set SAMPLE_DATE as index
Filtered_data = Filtered_data.set_index(
    "SAMPLE_DATE"
)

# EXTRACT REFLECTANCE SPECTRA: 400–700 nm
reflection_columns = [
    f"Refl{wavelength}nm"
    for wavelength in range(400, 701)
]

Reflection = Filtered_data[
    reflection_columns
].copy()

# RENAME REFLECTANCE COLUMNS TO WAVELENGTHS
column_mapping = {
    column: column.replace("Refl", "").replace("nm", "")
    for column in Reflection.columns
}

Reflection.rename(
    columns=column_mapping,
    inplace=True
)

# SEASONAL REFLECTANCE SPECTRA
# Convert reflectance data from wide format to long format
melted_df = Reflection.reset_index().melt(
    id_vars=["SAMPLE_DATE"],
    var_name="Wavebands",
    value_name="Reflectance"
)

# Convert wavelength to numeric
melted_df["Wavebands"] = pd.to_numeric(
    melted_df["Wavebands"]
)

# Assign observations to seasons
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return np.nan

# Add month information
melted_df["MONTH"] = (
    melted_df["SAMPLE_DATE"].dt.month
)

# Assign seasons
melted_df["Season"] = (
    melted_df["MONTH"].apply(get_season)
)

# Remove winter observations
melted_df = melted_df.dropna(
    subset=["Season"]
)
# PLOT SEASONAL REFLECTANCE SPECTRA
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=melted_df,
    x="Wavebands",
    y="Reflectance",
    hue="Season",
    errorbar="ci",
    err_style="band"
)
plt.xlabel(
    "Wavelength (nm)",
    fontsize=12
)
plt.ylabel(
    "Reflectance Spectra (sr$^{-1}$)",
    fontsize=12
)
plt.xticks(
    np.arange(400, 701, 20),
    rotation=90
)
plt.xticks(fontsize=11)
plt.legend()
plt.tight_layout()
plt.show()

# CALCULATE MEAN NATURAL ALGAL POPULATION REFLECTANCE
average_of_all_columns = (
    Reflection.mean(numeric_only=True)
)
average_of_all_columns.index.name = "Wavebands"
average = (
    average_of_all_columns
    .reset_index()
)
average.columns = [
    "Wavebands",
    "Reflectance"
]

# Convert wavelength to numeric
average["Wavebands"] = pd.to_numeric(
    average["Wavebands"]
)
# Rename reflectance column
finalRef = average.rename(
    columns={
        "Reflectance": "Natural_Population"
    }).copy()

# CALCULATE TAXONOMIC BIOMASS PERCENTAGE
taxonomic_columns = [
    "diatom",
    "dino1",
    "hapto6",
    "chloro"
]

# Remove samples where HPLCHLA is zero or missing
valid_taxa_data = Filtered_data[
    Filtered_data["HPLCHLA"] > 0
].copy()

# Calculate taxonomic biomass fraction
taxa_biomass = (
    valid_taxa_data[taxonomic_columns]
    .div(
        valid_taxa_data["HPLCHLA"],
        axis=0
    )
)

# Convert fractions to percentages
taxa_percent = taxa_biomass * 100

# RENAME TAXONOMIC PERCENTAGE COLUMNS
taxa_percent.rename(
    columns={
        "diatom": "diatom_percent",
        "dino1": "dino_percent",
        "hapto6": "hapto_percent",
        "chloro": "chloro_percent"
    },
    inplace=True
)

# COMBINE REFLECTANCE AND TAXONOMIC DATA
Reflection_taxa = pd.concat(
    [
        Reflection.loc[valid_taxa_data.index],
        taxa_percent
    ],
    axis=1
)

# SELECT SAMPLES WITH >80% TAXONOMIC CONTRIBUTION
# Diatoms
Diatoms = (Reflection_taxa[
        Reflection_taxa["diatom_percent"] > 80]
    .mean(numeric_only=True))

Diatoms.index.name = "Wavebands"
Diatoms = Diatoms.rename("Diatoms")
Diatoms1 = (
    Diatoms
    .reset_index()
)

# Dinoflagellates
Dino = (Reflection_taxa[
        Reflection_taxa["dino_percent"] > 80]
    .mean(numeric_only=True))

Dino.index.name = "Wavebands"
Dino = Dino.rename(
    "Dinoflagellates"
)

Dino1 = (
    Dino
    .reset_index()
)
# Haptophytes
Hapto = (Reflection_taxa[
        Reflection_taxa["hapto_percent"] > 80]
    .mean(numeric_only=True))
Hapto.index.name = "Wavebands"
Hapto = Hapto.rename("Haptophytes")

Hapto1 = (Hapto.reset_index())

# Chlorophytes
Chloro = (
    Reflection_taxa[
        Reflection_taxa["chloro_percent"] > 80
    ]
    .mean(numeric_only=True)
)

Chloro.index.name = "Wavebands"
Chloro = Chloro.rename("Chlorophytes")
Chloro1 = (
    Chloro
    .reset_index()
)

# REMOVE TAXONOMIC PERCENTAGE ROWS
taxonomic_percentage_columns = [
    "diatom_percent",
    "dino_percent",
    "hapto_percent",
    "chloro_percent"
]

Diatoms1 = Diatoms1[
    ~Diatoms1["Wavebands"].isin(
        taxonomic_percentage_columns)]

Dino1 = Dino1[
    ~Dino1["Wavebands"].isin(
        taxonomic_percentage_columns)]

Hapto1 = Hapto1[
    ~Hapto1["Wavebands"].isin(
        taxonomic_percentage_columns)]

Chloro1 = Chloro1[
    ~Chloro1["Wavebands"].isin(
        taxonomic_percentage_columns)]

# COMBINE TAXONOMIC REFLECTANCE WITH NATURAL POPULATION
Reflection_4taxa = pd.concat(
    [Diatoms1,Dino1,Hapto1,Chloro1,finalRef],
    axis=1)
# Remove duplicate columns
Reflection_4taxa = Reflection_4taxa.loc[
    :, ~Reflection_4taxa.columns.duplicated()]

# REMOVE UNNECESSARY INDEX COLUMN
if "index" in Reflection_4taxa.columns:
    Reflection_4taxa = Reflection_4taxa.drop(
        columns=["index"]
    )

# PLOT REFLECTANCE SPECTRA OF MAJOR ALGAL TAXA
x = Reflection_4taxa["Wavebands"]
y = Reflection_4taxa[
    [
        "Diatoms",
        "Dinoflagellates",
        "Haptophytes",
        "Chlorophytes",
        "Natural_Population"
    ]
]
plt.figure(figsize=(7, 4))
plt.plot(x,y,linestyle="solid",
    label=["Diatoms","Dinoflagellates",
        "Haptophytes","Chlorophytes",
        "Natural Population"])
plt.xlabel("Wavelength (nm)",fontsize=12)
plt.ylabel("Reflectance Spectra (sr$^{-1}$)",fontsize=12)
plt.xticks(np.arange(400, 701, 20),
    rotation=90)
plt.xticks(fontsize=11)
plt.legend()
plt.tight_layout()
plt.show()
