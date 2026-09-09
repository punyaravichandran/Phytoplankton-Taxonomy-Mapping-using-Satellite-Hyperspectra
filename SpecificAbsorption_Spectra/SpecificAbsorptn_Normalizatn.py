"""
Specific Absorption Normalization
Author: Punya
"""

# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# LOAD DATA
file_path = (
    "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
)

df = pd.read_csv(file_path)
# CONVERT DATE
df["SAMPLE_DATE"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)
# FILTER DATA FOR DEPTH < 20 m
df = df[
    df["DEPTH"] < 20
].copy()
# EXTRACT MONTH
df["Month"] = df["SAMPLE_DATE"].dt.month
# CALCULATE MONTHLY MEAN
monthly_mean = (
    df.groupby("Month")
    .mean(numeric_only=True)
)

# ABSORPTION WAVELENGTHS: 400–700 nm
absorption_columns = [
    f"wv{wavelength}nm"
    for wavelength in range(400, 701)
]

Abs_Col = df[
    absorption_columns
].copy()

# CALCULATE SPECIFIC ABSORPTION
# Remove samples with zero or missing HPLCHLA
valid_mask = (
    df["HPLCHLA"] > 0
) & (
    df["HPLCHLA"].notna()
)

df_valid = df.loc[
    valid_mask
].copy()

Abs_Col_valid = Abs_Col.loc[
    valid_mask
].copy()

Specific_Ab = Abs_Col_valid.div(
    df_valid["HPLCHLA"],
    axis=0
)

# ADD TAXONOMIC DATA
Specific_Ab["diatom"] = df_valid["diatom"]
Specific_Ab["dino1"] = df_valid["dino1"]
Specific_Ab["hapto6"] = df_valid["hapto6"]
Specific_Ab["dictyo"] = df_valid["dictyo"]
Specific_Ab["chloro"] = df_valid["chloro"]
Specific_Ab["HPLCHLA"] = df_valid["HPLCHLA"]

# CALCULATE TAXONOMIC CONTRIBUTION (%)
Specific_Ab["diatom_percent"] = (
    Specific_Ab["diatom"]
    / Specific_Ab["HPLCHLA"]
) * 100

Specific_Ab["dino_percent"] = (
    Specific_Ab["dino1"]
    / Specific_Ab["HPLCHLA"]
) * 100

Specific_Ab["hapto_percent"] = (
    Specific_Ab["hapto6"]
    / Specific_Ab["HPLCHLA"]
) * 100

Specific_Ab["dictyo_percent"] = (
    Specific_Ab["dictyo"]
    / Specific_Ab["HPLCHLA"]
) * 100

Specific_Ab["chloro_percent"] = (
    Specific_Ab["chloro"]
    / Specific_Ab["HPLCHLA"]
) * 100

# DIATOM-DOMINANT SAMPLES (>80%)
Specific_AbCol = Specific_Ab[
    Specific_Ab["diatom_percent"] > 80
].copy()

Diatoms = (
    Specific_AbCol[absorption_columns]
    .mean()
    .rename("Diatoms")
)
# DINOPHYCEAE-DOMINANT SAMPLES (>80%)
Specific_AbCol1 = Specific_Ab[
    Specific_Ab["dino_percent"] > 80
].copy()

Dino = (
    Specific_AbCol1[absorption_columns]
    .mean()
    .rename("Dinoflagellates")
)
# HAPTOPHYTE-DOMINANT SAMPLES (>80%)
Specific_AbCol2 = Specific_Ab[
    Specific_Ab["hapto_percent"] > 80
].copy()

Hapto = (
    Specific_AbCol2[absorption_columns]
    .mean()
    .rename("Haptophytes")
)
# DICTYOPHYTE-DOMINANT SAMPLES (>80%)
Specific_AbCol3 = Specific_Ab[
    Specific_Ab["dictyo_percent"] > 80
].copy()

Dictyo = (
    Specific_AbCol3[absorption_columns]
    .mean()
    .rename("Dictyophytes")
)
# CHLOROPHYTE-DOMINANT SAMPLES (>80%)
Specific_AbCol4 = Specific_Ab[
    Specific_Ab["chloro_percent"] > 80
].copy()

Chloro = (
    Specific_AbCol4[absorption_columns]
    .mean()
    .rename("Chlorophytes")
)
# COMBINE TAXON-SPECIFIC SPECTRA
Spec_abs_5taxa = pd.concat(
    [
        Diatoms,
        Dino,
        Hapto,
        Dictyo,
        Chloro
    ],
    axis=1
)
# CONVERT WAVELENGTH INDEX TO NUMERIC VALUES
wavelengths = np.arange(
    400,
    701
)

Spec_abs_5taxa.index = wavelengths
Spec_abs_5taxa.index.name = (
    "Wavelength (nm)"
)

# MONTHLY SPECIFIC ABSORPTION
Specific_Ab_monthly = (
    Specific_Ab[absorption_columns]
    .copy()
)

Specific_Ab_monthly["Month"] = (
    df_valid["Month"].values
)

# Calculate monthly mean specific absorption
Specific_AbRow = (
    Specific_Ab_monthly
    .groupby("Month")
    .mean()
)
# SELECT REQUIRED MONTHS
selected_months = [
    3, 4, 5, 6, 7, 9, 10
]

Specific_AbRow1 = (
    Specific_AbRow
    .loc[selected_months]
    .T
)

Specific_AbRow1.columns = [
    "March",
    "April",
    "May",
    "June",
    "July",
    "September",
    "October"
]

Specific_AbRow1.index = wavelengths

Specific_AbRow1.index.name = (
    "Wavelength (nm)"
)

# NORMAL ALGAL POPULATION SPECTRUM
Normal_Population = (
    Specific_AbRow1
    .mean(axis=1)
    .rename("Normal_Population")
)

# COMBINE TAXA + NORMAL POPULATION
Spec_abs_4taxa = pd.concat(
    [
        Diatoms,
        Dino,
        Hapto,
        Chloro,
        Normal_Population
    ],
    axis=1
)

Spec_abs_4taxa.index = wavelengths

Spec_abs_4taxa.index.name = (
    "Wavelength (nm)"
)
# NORMALIZE AT 440 nm
# Select 440 nm explicitly
Filter_440 = Spec_abs_4taxa.loc[
    440
]

# NORMALIZE SPECIFIC ABSORPTION
spectra_columns = [
    "Diatoms",
    "Dinoflagellates",
    "Haptophytes",
    "Chlorophytes",
    "Normal_Population"
]

y = Spec_abs_4taxa[
    spectra_columns
].divide(
    Filter_440[spectra_columns],
    axis=1
)

y.index.name = (
    "Wavelength (nm)"
)

# PLOT NORMALIZED SPECIFIC ABSORPTION
plt.figure(
    figsize=(12, 7)
)
plt.plot(
    y.index,
    y["Diatoms"],
    label="Diatoms",
    linewidth=2
)
plt.plot(
    y.index,
    y["Dinoflagellates"],
    label="Dinoflagellates",
    linewidth=2
)
plt.plot(
    y.index,
    y["Haptophytes"],
    label="Haptophytes",
    linewidth=2
)
plt.plot(
    y.index,
    y["Chlorophytes"],
    label="Chlorophytes",
    linewidth=2
)
plt.plot(
    y.index,
    y["Normal_Population"],
    label="Normal Population",
    linewidth=3
)
plt.xlabel(
    "Wavelength (nm)",
    fontsize=12
)

plt.ylabel(
    "Normalized Specific Absorption",
    fontsize=12
)

plt.xticks(
    np.arange(400, 701, 20),
    rotation=90
)

plt.legend()
plt.grid(
    True,
    alpha=0.3
)
plt.tight_layout()
plt.show()
