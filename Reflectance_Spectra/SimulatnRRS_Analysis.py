"""
Simulation reflectance of Phytoplankton Taxa & its Seasonal variability
Author: Punya
"""

# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as ticker
import datetime

# Load Data
SAMRef_data = pd.read_csv("Reflectance_SAM_2019-2024_cdom160.csv")
# Pigment, taxonomy, and absorption data
Absorption_data = pd.read_csv("Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv")

# Rename the SAMPLE_ID column in the reflectance dataset
SAMRef_data.rename(columns={"isdata$SAMPLE_ID": "SAMPLE_ID"},inplace=True)
# Merge Reflectance and Absorption Data
Final_df = pd.merge(SAMRef_data,Absorption_data,on="SAMPLE_ID")
Final_df.head()

# Calculate Mean Natural Algal Population Reflectance
average_of_all_columns = SAMRef_data.mean( numeric_only=True)
average_of_all_columns.index.name = "Wavebands"
average = average_of_all_columns.reset_index()
# Rename columns
average.columns = ["Wavebands","Reflectance"]
# Remove SAMPLE_ID from the wavelength data
finalRef = average[average["Wavebands"] != "SAMPLE_ID"].copy()

# Convert Date Information
Final_df["SAMPLE_DATE"] = pd.to_datetime(Final_df["Date"],format="%d-%m-%Y")
# Extract year, month, and day
Final_df["YEAR"] = Final_df["SAMPLE_DATE"].dt.year
Final_df["MONTH"] = Final_df["SAMPLE_DATE"].dt.month
Final_df["DAY"] = Final_df["SAMPLE_DATE"].dt.day

# Filter Data by Depth
Filtered_data = Final_df[Final_df["DEPTH"] < 20].copy()
# Set SAMPLE_DATE as the index
Filtered_data = Filtered_data.set_index("SAMPLE_DATE")

# Extract Reflectance Spectra (400-700 nm)
reflection_columns = [f"Refl{wavelength}nm"for wavelength in range(400, 701)]
Reflection = Filtered_data[reflection_columns].copy()

# Rename Reflectance Columns to Wavelengths
column_mapping = {column: column.replace("Refl", "").replace("nm", "")
    for column in Reflection.columns
}
Reflection = Reflection.rename(columns=column_mapping)

# Transpose Reflectance Data
Reflection_Coloumn = Reflection.transpose()
Reflection_Coloumn.index.name = "Wavebands"
# Reset index to convert wavelengths into a column
Reflection_Coloumn = Reflection_Coloumn.reset_index()
Reflection_Coloumn.head()

# Seasonal Reflectance Spectra
plt.rcParams["figure.figsize"] = (8, 4)
value_vars = [
    column
    for column in Reflection_Coloumn.columns
    if column != "Wavebands"
]
# Assign observations to seasons
melted_df["Season"] = pd.cut(
    melted_df["Date"].dt.month,
    bins=[3, 5, 8, 11],
    labels=["Spring", "Summer", "Fall"],
    ordered=False,
    right=True
)
# Remove observations that were not assigned to a season
melted_df = melted_df.dropna(subset=["Season"])

# Plot Seasonal Reflectance Spectra
sns.lineplot(x="Wavebands",y="Reflection",data=melted_df,hue="Season",errorbar="ci",err_style="band")
plt.xlabel("Wavelength (nm)",fontsize=12)
plt.ylabel("Reflectance spectra (sr-1)",fontsize=12)
plt.xticks(np.arange(0, 301, step=10),rotation=90)
plt.rcParams["xtick.labelsize"] = 12
plt.legend()
plt.tight_layout()
plt.show()

# Calculate Taxonomic Biomass Percentage
taxa_biomass = Filtered_data[["diatom","dino1","hapto6","chloro"]].div(Filtered_data["HPLCHLA"],axis=0)
# Convert fractions to percentages
taxa_percent = taxa_biomass * 100
# Rename Taxonomic Percentage Columns
taxa_percent = taxa_percent.rename(
    columns={
        "diatom": "diatom_percent",
        "dino1": "dino_percent",
        "hapto6": "hapto_percent",
        "chloro": "chloro_percent"
    }
)
# Combine Reflectance and Taxonomic Data
Reflection_taxa = pd.concat([Reflection,taxa_percent],axis=1)
# Select observations where diatoms contribute >80%
Diatoms = Reflection_taxa[Reflection_taxa["diatom_percent"] > 80].mean()
Diatoms.index.name = "Wavebands"
Diatoms = Diatoms.rename("Diatoms")
Diatoms1 = Diatoms.reset_index()
# Remove taxonomic percentage rows
Diatoms1 = Diatoms1.drop(Diatoms1[Diatoms1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# Select observations where dinoflagellates contribute >80%
Dino = Reflection_taxa(Reflection_taxa["dino_percent"] > 80].mean()
Dino.index.name = "Wavebands"
Dino = Dino.rename("Dinoflagellates")
Dino1 = Dino.reset_index()
Dino1 = Dino1.drop(Dino1[Dino1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# Select observations where haptophytes contribute >80%
Hapto = Reflection_taxa[Reflection_taxa["hapto_percent"] > 80].mean()
Hapto.index.name = "Wavebands"
Hapto = Hapto.rename("Haptophytes")
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1.drop(Hapto1[Hapto1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# Select observations where chlorophytes contribute >80%
Chloro = Reflection_taxa[Reflection_taxa["chloro_percent"] > 80].mean()
Chloro.index.name = "Wavebands"
Chloro = Chloro.rename("Chlorophytes")
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1.drop(Chloro1[Chloro1["Wavebands"].isin(["diatom_percent","dino_percent","hapto_percent","chloro_percent"])].index)
# Combine Taxa Reflectance with Natural Population Reflectance
finalRef = finalRef.rename(columns={"Reflectance": "Natural_Population"})
Mixed_Spectra = finalRef["Natural_Population"].reset_index()
# Combine all taxonomic reflectance spectra
Reflection_4taxa = pd.concat([Diatoms1,Dino1,Hapto1,Chloro1,Mixed_Spectra],axis=1)
# Remove duplicate columns
Reflection_4taxa = Reflection_4taxa.loc[:,~Reflection_4taxa.columns.duplicated()]
# Remove unnecessary index column
Reflection_4taxa = Reflection_4taxa.drop(columns=["index"])
Reflection_4taxa.head()

# Plot Reflectance Spectra of Major Algal Taxa
x = Reflection_4taxa["Wavebands"]
y = Reflection_4taxa[["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Natural_Population"]]
plt.rcParams["figure.figsize"] = (7, 4)
plt.plot(x,y,linestyle="solid",label=["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Natural Population"])
plt.xlabel("Wavelength (nm)",fontsize=12)
plt.ylabel("Reflectance Spectra (sr-1)",fontsize=12)
plt.xticks(np.arange(0, 301, step=10))
plt.rcParams["xtick.labelsize"] = 11
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
