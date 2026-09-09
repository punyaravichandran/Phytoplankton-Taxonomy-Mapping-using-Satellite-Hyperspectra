"""
Specific Absorption Normalization
Author: Punya
"""

# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# LOAD DATA
file_path = "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
df = pd.read_csv(file_path)

# CONVERT DATE
df["SAMPLE_DATE"] = pd.to_datetime(df["Date"], errors="coerce")

# FILTER DATA FOR DEPTH < 20 m
df = df[df["DEPTH"] < 20].copy()

# CALCULATE MONTHLY MEAN
df["Month"] = df["SAMPLE_DATE"].dt.month
monthly_mean = (df.groupby("Month").mean(numeric_only=True))

# ABSORPTION WAVELENGTHS: 400–700 nm
absorption_columns = [
    f"wv{wavelength}nm"
    for wavelength in range(400, 701)
]
Abs_Col = df[absorption_columns].copy()

# CALCULATE SPECIFIC ABSORPTION
Specific_Ab = Abs_Col.div(df["HPLCHLA"],axis=0)
Specific_Ab["diatom"] = df["diatom"]
Specific_Ab["dino1"] = df["dino1"]
Specific_Ab["hapto6"] = df["hapto6"]
Specific_Ab["dictyo"] = df["dictyo"]
Specific_Ab["chloro"] = df["chloro"]
Specific_Ab["HPLCHLA"] = df["HPLCHLA"]

# CALCULATE TAXONOMIC CONTRIBUTION (%)
Specific_Ab["diatom_percent"] = (Specific_Ab["diatom"] /Specific_Ab["HPLCHLA"]) * 100
Specific_Ab["dino_percent"] = (Specific_Ab["dino1"] /Specific_Ab["HPLCHLA"]) * 100
Specific_Ab["hapto_percent"] = (Specific_Ab["hapto6"] /Specific_Ab["HPLCHLA"]) * 100
Specific_Ab["dictyo_percent"] = (Specific_Ab["dictyo"] /Specific_Ab["HPLCHLA"]) * 100
Specific_Ab["chloro_percent"] = (Specific_Ab["chloro"] /Specific_Ab["HPLCHLA"]) * 100

# DIATOM-DOMINANT SAMPLES (>80%)
Specific_AbCol = Specific_Ab[Specific_Ab["diatom_percent"] > 80].copy()
Diatoms = Specific_AbCol[absorption_columns].mean()
Diatoms.index.name = "Wavebands"
Diatoms = Diatoms.rename("Diatoms")
# DINOPHYCEAE-DOMINANT SAMPLES (>80%)
Specific_AbCol1 = Specific_Ab[Specific_Ab["dino_percent"] > 80].copy()
Dino = Specific_AbCol1[absorption_columns].mean()
Dino.index.name = "Wavebands"
Dino = Dino.rename("Dinoflagellates")
# HAPTOPHYTE-DOMINANT SAMPLES (>80%)
Specific_AbCol2 = Specific_Ab[Specific_Ab["hapto_percent"] > 80].copy()
Hapto = Specific_AbCol2[absorption_columns].mean()
Hapto.index.name = "Wavebands"
Hapto = Hapto.rename("Haptophytes")
# DICTYOPHYTE-DOMINANT SAMPLES (>80%)
Dictyo = Specific_AbCol2[Specific_AbCol2["dictyo_percent"] > 80][absorption_columns].mean()
Dictyo.index.name = "Wavebands"
Dictyo = Dictyo.rename("Dictyophytes")
# CHLOROPHYTE-DOMINANT SAMPLES (>80%)
Specific_AbCol3 = Specific_Ab[Specific_Ab["chloro_percent"] > 80].copy()
Chloro = Specific_AbCol3[absorption_columns].mean()
Chloro.index.name = "Wavebands"
Chloro = Chloro.rename("Chlorophytes")

# COMBINE TAXON-SPECIFIC SPECTRA
Spec_abs_5taxa = pd.concat([Diatoms,Dino,Hapto,Dictyo,Chloro],axis=1)
# MONTHLY SPECIFIC ABSORPTION
Specific_Ab_monthly = Specific_Ab[absorption_columns].copy()
Specific_Ab_monthly["Month"] = df["Month"].values
Specific_AbRow = (Specific_Ab_monthly.groupby("Month").mean())
# SELECT REQUIRED MONTHS
Specific_AbRow1 = Specific_AbRow.loc[
    Specific_AbRow.index.isin(
        [3, 4, 5, 6, 7, 9, 10]
    )
].T
Specific_AbRow1.columns = ["March","April","May","June","July","September","October"]
Specific_AbRow1.index.name = "Wavebands"

# NORMAL ALGAL POPULATION SPECTRUM
Normal_Population = Specific_AbRow1.mean(axis=1)
Normal_Population = Normal_Population.rename("Normal_Population")
# COMBINE TAXA + NORMAL POPULATION
Spec_abs_4taxa = pd.concat([Diatoms,Dino,Hapto,Chloro,Normal_Population],axis=1)

# NORMALIZE AT 440 nm
Filter_440 = Spec_abs_4taxa.iloc[40]
y = Spec_abs_4taxa[["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Normal_Population"]]
    .divide(Filter_440[["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Normal_Population"]],axis=1)
y.index = range(400, 701)
y.index.name = "Wavelength (nm)"

# PLOT NORMALIZED SPECIFIC ABSORPTION
plt.figure(figsize=(12, 7))
plt.plot(y.index,y["Diatoms"],label="Diatoms",linewidth=2)
plt.plot(y.index,y["Dinoflagellates"],label="Dinoflagellates",linewidth=2)
plt.plot(y.index,y["Haptophytes"],label="Haptophytes",linewidth=2)
plt.plot(y.index,y["Chlorophytes"],label="Chlorophytes",linewidth=2)
plt.plot(y.index,y["Normal_Population"],label="Normal Population",linewidth=3)
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Normalized Specific Absorption", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
