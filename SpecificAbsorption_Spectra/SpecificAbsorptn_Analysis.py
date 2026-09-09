"""
Seasonal & Algal Taxa Specific Absorption Analysis 
"""

# 1. Load Required Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# Load In-situ Data
Insitu_data = pd.read_csv("Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv")

# Convert Date Information
Insitu_data["SAMPLE_DATE"] = pd.to_datetime(Insitu_data["Date"],format="%d-%m-%Y",errors="coerce")
# Extract year, month, and day
Insitu_data["YEAR"] = Insitu_data["SAMPLE_DATE"].dt.year
Insitu_data["MONTH"] = Insitu_data["SAMPLE_DATE"].dt.month
Insitu_data["DAY"] = Insitu_data["SAMPLE_DATE"].dt.day
#Select Data for Analysis
Filtered_data = Insitu_data.copy()
# Set SAMPLE_DATE as Index
HL_data_day = Filtered_data.set_index("SAMPLE_DATE")
# Extract Absorption Spectra (400-700 nm)
absorption_columns = [f"wv{wavelength}nm"
    for wavelength in range(400, 701)
]
# Select absorption bands
Absorption = HL_data_day[absorption_columns].copy()
# Rename Absorption Columns to Wavelengths
column_mapping = {
    column: column.replace("wv", "").replace("nm", "")
    for column in Absorption.columns
}
Absorption = Absorption.rename(columns=column_mapping)

# Calculate Specific Absorption
Specific_AbCol = Absorption.div(HL_data_day["HPLCHLA"],axis=0)
# Transpose Specific Absorption Data
Specific_AbRow = Specific_AbCol.transpose()
# Set wavelength index name
Specific_AbRow.index.name = "Wavebands"
# Reset index to convert wavelength into a column
Specific_AbRow1 = Specific_AbRow.reset_index()

# Seasonal Specific Absorption Spectra
plt.rcParams["figure.figsize"] = (8, 4)
# Get all columns except Wavebands
value_vars = [
    column
    for column in Specific_AbRow1.columns
    if column != "Wavebands"
]
# Convert dataframe to long format
melted_df = pd.melt(Specific_AbRow1,id_vars=["Wavebands"],value_vars=value_vars,var_name="Date",value_name="Specific Absorption")
# Convert Date to datetime
melted_df["Date"] = pd.to_datetime(melted_df["Date"])
# Assign Observations to Seasons
melted_df["Season"] = pd.cut(melted_df["Date"].dt.month,bins=[3, 5, 8, 11],labels=["Spring","Summer","Fall"],
    ordered=False,
    right=True)
# Remove observations without a season
melted_df = melted_df.dropna(subset=["Season"])

# Plot Seasonal Specific Absorption Spectra
sns.lineplot(x="Wavebands",y="Specific Absorption",data=melted_df,hue="Season",errorbar="ci",err_style="band")
plt.xlabel("Wavelength (nm)",fontsize=11)
plt.ylabel("Specific absorption of\nphytoplankton population (m-1)",fontsize=11)
plt.xticks(np.arange(0, 301, step=10),rotation=90)
plt.rcParams["xtick.labelsize"] = 10
plt.legend()
plt.tight_layout()
plt.show()

# Calculate Taxonomic Biomass Percentage
taxa_biomass = Filtered_data[["diatom","dino1","hapto6","dictyo","chloro"]].div(Filtered_data["HPLCHLA"],axis=0)
# Convert fractional contribution to percentage
taxa_percent = taxa_biomass * 100
# Rename Taxonomic Percentage Columns
taxa_percent = taxa_percent.rename(
    columns={
        "diatom": "diatom_percent",
        "dino1": "dino_percent",
        "hapto6": "hapto_percent",
        "dictyo": "dictyo_percent",
        "chloro": "chloro_percent"
    }
)
# Combine Specific Absorption and Taxonomic Data
# Reset index before concatenation to ensure matching row positions
Specific_AbCol_reset = Specific_AbCol.reset_index(drop=True)
Specific_AbCol2 = pd.concat([Specific_AbCol_reset,taxa_percent.reset_index(drop=True)],axis=1)
# Identify Dominant Diatom Samples
# Select observations where diatoms contribute >80%
Diatoms = Specific_AbCol2[Specific_AbCol2["diatom_percent"] > 80].mean()
Diatoms.index.name = "Wavebands"
Diatoms = Diatoms.rename("Diatoms")
Diatoms1 = Diatoms.reset_index()
percentage_columns = ["diatom_percent","dino_percent","hapto_percent","dictyo_percent","chloro_percent"]
Diatoms1 = Diatoms1.drop(Diatoms1[Diatoms1["Wavebands"].isin(percentage_columns)].index)
# Identify Dominant Dinoflagellate Samples
# Select observations where dinoflagellates contribute >80%
Dino = Specific_AbCol2[Specific_AbCol2["dino_percent"] > 80].mean()
Dino.index.name = "Wavebands"
Dino = Dino.rename("Dinoflagellates")
Dino1 = Dino.reset_index()
Dino1 = Dino1.drop(Dino1[Dino1["Wavebands"].isin(percentage_columns)].index)
# Identify Dominant Haptophyte Samples
# Select observations where haptophytes contribute >80%
Hapto = Specific_AbCol2[Specific_AbCol2["hapto_percent"] > 80].mean()
Hapto.index.name = "Wavebands"
Hapto = Hapto.rename("Haptophytes")
Hapto1 = Hapto.reset_index()
Hapto1 = Hapto1.drop(Hapto1[Hapto1["Wavebands"].isin(percentage_columns)].index)
# Identify Dominant Dictyophyte Samples
# Select observations where dictyophytes contribute >80%
Dictyo = Specific_AbCol2[Specific_AbCol2["dictyo_percent"] > 80].mean()
Dictyo.index.name = "Wavebands"
Dictyo = Dictyo.rename("Dictyophytes")
Dictyo1 = Dictyo.reset_index()
Dictyo1 = Dictyo1.drop(Dictyo1[Dictyo1["Wavebands"].isin(percentage_columns)].index)
# Identify Dominant Chlorophyte Samples
# Select observations where chlorophytes contribute >80%
Chloro = Specific_AbCol2[Specific_AbCol2["chloro_percent"] > 80].mean()
Chloro.index.name = "Wavebands"
Chloro = Chloro.rename("Chlorophytes")
Chloro1 = Chloro.reset_index()
Chloro1 = Chloro1.drop(Chloro1[Chloro1["Wavebands"].isin(percentage_columns)].index)
# Combine Dominant Taxa
combined_taxa = pd.merge(Diatoms1,Dino1,on="Wavebands",how="left")
combined_taxa1 = pd.merge(combined_taxa,Hapto1,on="Wavebands",how="left")
combined_taxa2 = pd.merge(combined_taxa1,Chloro1,on="Wavebands",how="left")
# Calculate Mixed Population Specific Absorption
# Sum the specific absorption values across the taxonomic groups.
combined_taxa2["Mixed_Population"] = (combined_taxa2.drop(columns=["Wavebands"]).sum(axis=1))
# Select wavelength and mixed population columns
Mixed_population = combined_taxa2[["Wavebands","Mixed_Population"]]
# Combine Individual Taxa with Mixed Population
combine_data = pd.merge(Mixed_population,Diatoms1,on="Wavebands",how="left")
combine_data1 = pd.merge(combine_data,Dino1,on="Wavebands",how="left")
combine_data2 = pd.merge(combine_data1,Hapto1,on="Wavebands",how="left")
combine_data3 = pd.merge(combine_data2,Chloro1,on="Wavebands",how="left")
combine_data3.head()

# Plot Specific Absorption Spectra of Algal Taxa
x = combine_data3["Wavebands"]
y = combine_data3[["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Mixed_Population"]]
plt.rcParams["figure.figsize"] = (10, 5)
plt.plot(x,y,linestyle="solid",label=["Diatoms","Dinoflagellates","Haptophytes","Chlorophytes","Mixed_Population"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Specific Absorption Spectra of Algal Taxa")
plt.xticks(np.arange(0, 301, step=10))
plt.rcParams["xtick.labelsize"] = 7
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
