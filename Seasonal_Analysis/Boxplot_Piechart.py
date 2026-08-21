"""
Scotian Shelf Phytoplankton Taxonomy - Seasonal Analysis
Author: Punya
"""

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as ticker

# LOAD DATA
file_path = "Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
Insitu_data = pd.read_csv(file_path)

# DATE PROCESSING
Insitu_data["SAMPLE_DATE"] = pd.to_datetime(Insitu_data["Date"],format="%d-%m-%Y")

# Extract year, month, and day
Insitu_data["YEAR"] = Insitu_data["SAMPLE_DATE"].dt.year
Insitu_data["MONTH"] = Insitu_data["SAMPLE_DATE"].dt.month
Insitu_data["DAY"] = Insitu_data["SAMPLE_DATE"].dt.day

# FILTER DATA (DEPTH < 20m)
Filtered_data = Insitu_data[Insitu_data["DEPTH"] < 20].copy()

# MONTHLY SAMPLE COUNT
HL_data_month = Filtered_data.set_index("MONTH")
monthly_sample_counts = (Filtered_data["MONTH"].value_counts().sort_index())
display(monthly_sample_counts)

# DEFINE SEASONS
Spring_Data = HL_data_month[HL_data_month.index.isin([3, 4, 5])]
Summer_Data = HL_data_month[HL_data_month.index.isin([6, 7])]
Fall_Data = HL_data_month[HL_data_month.index.isin([9, 10])]

# Select Phytoplankton Taxonomic Groups
selected_columns = ["diatom","dino1", "hapto6", "dictyo","chloro"]

# Replace zero concentrations with NaN before log transformation
Springdata = Spring_Data[selected_columns].replace(0, np.nan)
Summerdata = Summer_Data[selected_columns].replace(0, np.nan)
Falldata = Fall_Data[selected_columns].replace(0, np.nan)

# Apply log10 transformation
Springdata1 = np.log10(Springdata)
Summerdata1 = np.log10(Summerdata)
Falldata1 = np.log10(Falldata)

# Preserve NaN values
Springdata2 = Springdata1.fillna(Springdata1)
Summerdata2 = Summerdata1.fillna(Summerdata1)
Falldata2 = Falldata1.fillna(Falldata1)


# BOXPLOT
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# Spring
sns.boxplot(data=Springdata2, showmeans=True, meanprops={"markerfacecolor": "white"}, ax=axs[0])
axs[0].set_ylabel("Concentration (microgram/L)")
axs[0].set_title("Spring")
# Summer
sns.boxplot( data=Summerdata2, showmeans=True, meanprops={"markerfacecolor": "white"}, ax=axs[1])
axs[1].set_title("Summer")
# Fall
sns.boxplot(data=Falldata2,showmeans=True,meanprops={"markerfacecolor": "white"},ax=axs[2])
axs[2].set_title("Fall")
# Format Y-axis to Display Original Concentration
def real_concentration_formatter(y, _):  
    value = 10 ** y
    if value == 0:
        return f"{value:,.0f}"
    elif value >= 1:
        return f"{value:.0f}"
    else:
        return f"{value:.4f}"
for ax in axs:
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(real_concentration_formatter)
    )
plt.tight_layout()
plt.show()

# PIE CHARTS_ SEASONAL COMPOSITION
# 11. Seasonal Taxonomic Composition - Pie Charts
# =============================================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(12, 5)
)

# Labels for phytoplankton groups
labels = [
    "Diatom",
    "Dino",
    "Hapto",
    "Dictyo",
    "Chloro"
]


# -------------------------------------------------------------------------
# Spring
# -------------------------------------------------------------------------

sizes_spring = Spring_Data[selected_columns].sum()

axs[0].pie(
    sizes_spring,
    labels=labels,
    autopct="%1.0f%%",
    textprops={"size": "smaller"}
)

axs[0].set_title("Spring")


# -------------------------------------------------------------------------
# Summer
# -------------------------------------------------------------------------

sizes_summer = Summer_Data[selected_columns].sum()

axs[1].pie(
    sizes_summer,
    labels=labels,
    autopct="%1.0f%%",
    textprops={"size": "smaller"}
)

axs[1].set_title("Summer")


# -------------------------------------------------------------------------
# Fall
# -------------------------------------------------------------------------

sizes_fall = Fall_Data[selected_columns].sum()

axs[2].pie(
    sizes_fall,
    labels=labels,
    autopct="%1.0f%%",
    textprops={"size": "smaller"}
)

axs[2].set_title("Fall")


plt.tight_layout()
plt.show()
