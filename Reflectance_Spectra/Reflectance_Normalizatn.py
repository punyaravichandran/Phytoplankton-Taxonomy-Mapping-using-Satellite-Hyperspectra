"""
Simulated Reflectance Normalization
Author: Punya
"""

# Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# Load CSV files
SAMRef_data = pd.read_csv("Reflectance_SAM_2019-2024_cdom160.csv")
Absorption_data = pd.read_csv("Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv")

# Rename columns
SAMRef_data.rename(columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},inplace=True)
# Merge the two dataframes
Final_df = pd.merge(SAMRef_data,Absorption_data,on="SAMPLE_ID")
# Calculate average reflectance spectra
average_of_all_columns = SAMRef_data.mean(numeric_only=True)
# Set index name
average_of_all_columns.index.name = "Wavebands"
# Convert Series to DataFrame
average = average_of_all_columns.reset_index()
average.columns = ["Wavebands","Reflectance"]
# Remove SAMPLE_ID if present
finalRef = average[average["Wavebands"] != "SAMPLE_ID"].copy()
# Convert sample date to datetime
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"])
# Filter samples to depth < 20 m
Filtered_data = Final_df[Final_df["DEPTH"] < 20].copy()
# Select reflectance spectra from 400–700 nm
Reflection_columns = [
    f"Refl{wavelength}nm"
    for wavelength in range(400, 701)
]
Reflection = Filtered_data[Reflection_columns].copy()
# Rename reflectance columns
column_mapping = {
    column: column.replace("Refl", "").replace("nm", "")
    for column in Reflection.columns
}
Reflection.rename(columns=column_mapping,inplace=True)
# Transpose reflectance dataframe
Reflection_Column = Reflection.transpose()

# Calculate phytoplankton taxa percentage biomass
taxa_biomass = Filtered_data[["diatom", "dino1", "hapto6", "chloro"]].div(Filtered_data["HPLCHLA"],axis=0)
# Convert to percentage
taxa_percent = taxa_biomass * 100
# Rename columns
taxa_percent.rename(columns={
        "diatom": "diatom_percent",
        "dino1": "dino_percent",
        "hapto6": "hapto_percent",
        "chloro": "chloro_percent"
    },
    inplace=True
)
# Combine reflectance and taxa percentage data
Reflection_taxa = pd.concat([Reflection, taxa_percent],axis=1)

# DIATOMS
# Select samples where diatoms > 80%
Diatoms = Reflection_taxa[Reflection_taxa["diatom_percent"] > 80].mean()
Diatoms.index.name = "Wavebands"
Diatoms = Diatoms.rename("Diatoms")
# Convert Series to DataFrame
Diatoms1 = Diatoms.reset_index()
# Remove taxa percentage rows
Diatoms1 = Diatoms1.drop(Diatoms1[Diatoms1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# DINOFLAGELLATES
# Select samples where dinoflagellates > 80%
Dino = Reflection_taxa[Reflection_taxa["dino_percent"] > 80].mean()
Dino.index.name = "Wavebands"
Dino = Dino.rename("Dinoflagellates")
Dino1 = Dino.reset_index()
Dino1 = Dino1.drop(Dino1[Dino1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# HAPTOPHYTES
# Select samples where haptophytes > 80%
Hapto = Reflection_taxa[Reflection_taxa["hapto_percent"] > 80].mean()
Hapto.index.name = "Wavebands"
Hapto = Hapto.rename("Haptophytes")
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1.drop(Hapto1[Hapto1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# CHLOROPHYTES
# Select samples where chlorophytes > 80%
Chloro = Reflection_taxa[Reflection_taxa["chloro_percent"] > 80].mean()
Chloro.index.name = "Wavebands"
Chloro = Chloro.rename("Chlorophytes")
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1.drop(Chloro1[Chloro1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)

# Combine all taxa reflectance spectra
# Rename normal population reflectance
finalRef.rename(columns={"Reflectance": "Normal_Population"},inplace=True)
# Reset index
Mixed_Spectra = finalRef["Normal_Population"].reset_index()
# Combine all taxa
Reflection_4taxa = pd.concat([Diatoms1,Dino1,Hapto1,Chloro1,Mixed_Spectra],axis=1)
# Remove duplicated columns
Reflection_4taxa = Reflection_4taxa.loc[:,~Reflection_4taxa.columns.duplicated()]
# Remove unnecessary index column
if "index" in Reflection_4taxa.columns:
    Reflection_4taxa.drop(
        columns=["index"],
        inplace=True
    )
    
# Normalization using 440 nm reflectance
# 440 nm corresponds to row index 40
Filter_440 = Reflection_4taxa.iloc[40]
# Normalize spectra by 440 nm
x = Reflection_4taxa["Wavebands"]
spectra_columns = ["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Normal_Population"]
y = Reflection_4taxa[spectra_columns].divide(Filter_440[spectra_columns],axis=1)

# Plot normalized reflectance spectra
plt.figure(figsize=(7, 4))
plt.plot(x,y,linestyle="solid")
plt.xlabel("Wavelength (nm)",fontsize=12)
plt.ylabel("Normalized Reflectance (sr⁻¹)",fontsize=12)
# Set wavelength tick interval
plt.xticks(np.arange(0, len(x), step=10),x.iloc[::10],rotation=90)
plt.xticks(fontsize=10)
plt.legend(["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Normal Population"])
plt.tight_layout()
plt.show()
