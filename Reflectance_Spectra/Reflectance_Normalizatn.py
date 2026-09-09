"""
Simulated Reflectance Normalization
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
Absorption_data = pd.read_csv(
    "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)
# RENAME SAMPLE ID COLUMN
SAMRef_data.rename(
    columns={
        "isdata$SAMPLE_ID": "SAMPLE_ID"
    },
    inplace=True)

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
    format="%d-%m-%Y")

# FILTER SAMPLES: DEPTH < 20 m
Filtered_data = Final_df[
    Final_df["DEPTH"] < 20].copy()

# EXTRACT REFLECTANCE SPECTRA: 400–700 nm
Reflection_columns = [
    f"Refl{wavelength}nm"
    for wavelength in range(400, 701)
]

Reflection = Filtered_data[
    Reflection_columns
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

# Convert wavelength column names to integers
Reflection.columns = Reflection.columns.astype(int)
# CALCULATE NORMAL POPULATION REFLECTANCE
average_of_all_columns = Reflection.mean(
    numeric_only=True
)

average = (
    average_of_all_columns
    .rename_axis("Wavebands")
    .reset_index(name="Normal_Population")
)

average["Wavebands"] = average["Wavebands"].astype(int)

# CALCULATE PHYTOPLANKTON TAXONOMIC BIOMASS PERCENTAGE
taxonomic_columns = [
    "diatom",
    "dino1",
    "hapto6",
    "chloro"
]

# Keep samples with positive HPLCHLA
valid_taxa_data = Filtered_data[
    Filtered_data["HPLCHLA"] > 0
].copy()


# Calculate taxonomic biomass fractions
taxa_biomass = (
    valid_taxa_data[taxonomic_columns]
    .div(
        valid_taxa_data["HPLCHLA"],
        axis=0
    )
)

# Convert fractions to percentages
taxa_percent = taxa_biomass * 100
# RENAME TAXONOMIC COLUMNS
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


# TAXONOMIC REFLECTANCE: >80% DOMINANCE
taxa_percentage_columns = [
    "diatom_percent",
    "dino_percent",
    "hapto_percent",
    "chloro_percent"
]
# DIATOMS
Diatoms = (
    Reflection_taxa[
        Reflection_taxa["diatom_percent"] > 80
    ]
    .mean(numeric_only=True)
    .rename("Diatoms")
    .rename_axis("Wavebands")
    .reset_index()
)

Diatoms = Diatoms[
    ~Diatoms["Wavebands"].isin(
        taxa_percentage_columns
    )
]

# DINOFLAGELLATES
Dino = (
    Reflection_taxa[
        Reflection_taxa["dino_percent"] > 80
    ]
    .mean(numeric_only=True)
    .rename("Dinoflagellates")
    .rename_axis("Wavebands")
    .reset_index()
)

Dino = Dino[
    ~Dino["Wavebands"].isin(
        taxa_percentage_columns
    )
]

# HAPTOPHYTES
Hapto = (
    Reflection_taxa[
        Reflection_taxa["hapto_percent"] > 80
    ]
    .mean(numeric_only=True)
    .rename("Haptophytes")
    .rename_axis("Wavebands")
    .reset_index()
)

Hapto = Hapto[
    ~Hapto["Wavebands"].isin(
        taxa_percentage_columns
    )
]
# CHLOROPHYTES
Chloro = (
    Reflection_taxa[
        Reflection_taxa["chloro_percent"] > 80
    ]
    .mean(numeric_only=True)
    .rename("Chlorophytes")
    .rename_axis("Wavebands")
    .reset_index()
)

Chloro = Chloro[
    ~Chloro["Wavebands"].isin(
        taxa_percentage_columns
    )
]

# COMBINE ALL TAXONOMIC REFLECTANCE SPECTRA
Reflection_4taxa = pd.concat(
    [
        Diatoms,
        Dino,
        Hapto,
        Chloro,
        average
    ],
    axis=1
)

# REMOVE DUPLICATE COLUMNS
Reflection_4taxa = Reflection_4taxa.loc[
    :,
    ~Reflection_4taxa.columns.duplicated()
]

# Remove unnecessary index column if present
if "index" in Reflection_4taxa.columns:
    Reflection_4taxa.drop(
        columns=["index"],
        inplace=True
    )

# Convert wavelength to numeric
Reflection_4taxa["Wavebands"] = pd.to_numeric(
    Reflection_4taxa["Wavebands"]
)
# SORT BY WAVELENGTH
Reflection_4taxa.sort_values(
    by="Wavebands",
    inplace=True
)

Reflection_4taxa.reset_index(
    drop=True,
    inplace=True
)

# NORMALIZATION USING 440 nm REFLECTANCE
# Select the 440 nm reference reflectance
reference_row = Reflection_4taxa[
    Reflection_4taxa["Wavebands"] == 440
]

if reference_row.empty:
    raise ValueError(
        "440 nm wavelength was not found in Reflection_4taxa."
    )

Filter_440 = reference_row.iloc[0]

# Spectra to normalize
spectra_columns = [
    "Diatoms",
    "Dinoflagellates",
    "Haptophytes",
    "Chlorophytes",
    "Normal_Population"
]

# Normalize each spectrum by its 440 nm reflectance
Normalized_Spectra = (
    Reflection_4taxa[spectra_columns]
    .div(
        Filter_440[spectra_columns],
        axis=1
    )
)

# PLOT NORMALIZED REFLECTANCE SPECTRA
x = Reflection_4taxa["Wavebands"]
plt.figure(figsize=(7, 4))
plt.plot(
    x,
    Normalized_Spectra,
    linestyle="solid"
)

plt.xlabel(
    "Wavelength (nm)",
    fontsize=12
)

plt.ylabel(
    "Normalized Reflectance",
    fontsize=12
)

plt.xticks(
    np.arange(400, 701, 20),
    rotation=90
)

plt.xticks(fontsize=10)
plt.legend(
    [
        "Diatoms",
        "Dinoflagellates",
        "Haptophytes",
        "Chlorophytes",
        "Normal Population"
    ]
)
plt.tight_layout()
plt.show()
