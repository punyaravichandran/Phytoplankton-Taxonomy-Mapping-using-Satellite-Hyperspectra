"""
Phytoplankton Taxa Seasonal Analysis
Author: Punya
"""

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# LOAD DATA
file_path = "data/Scotian_Shelf_Pigment_Taxo_Absorption_2019-2024.csv"
df = pd.read_csv(file_path)

# DATE PROCESSING
df['SAMPLE_DATE'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['YEAR'] = df['SAMPLE_DATE'].dt.year
df['MONTH'] = df['SAMPLE_DATE'].dt.month
df['DAY'] = df['SAMPLE_DATE'].dt.day

# FILTER DATA (DEPTH < 20m)
df_shallow = df[df['DEPTH'] < 20].copy()

# MONTHLY SAMPLE COUNT
monthly_counts = df_shallow['MONTH'].value_counts().sort_index()
print("\nMonthly Sample Counts:\n", monthly_counts)

# DEFINE SEASONS
df_shallow = df_shallow.set_index('MONTH')

spring = df_shallow.loc[df_shallow.index.isin([3, 4, 5])]
summer = df_shallow.loc[df_shallow.index.isin([6, 7])]
fall   = df_shallow.loc[df_shallow.index.isin([9, 10])]
phyto_cols = ['diatom', 'dino1', 'hapto6', 'dictyo', 'chloro']

# BOXPLOT
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

def plot_box(data, ax, title):
    sns.boxplot(
        data=np.log10(data[phyto_cols] + 1e-9),
        showmeans=True,
        meanprops={"markerfacecolor": "white"},
        ax=ax
    )
    ax.set_title(title)

plot_box(spring, axs[0], "Spring")
plot_box(summer, axs[1], "Summer")
plot_box(fall, axs[2], "Fall")

# Convert log scale back to real values
def real_formatter(y, _):
    val = 10 ** y
    if val >= 1:
        return f"{val:.1f}"
    else:
        return f"{val:.3f}"

for ax in axs:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(real_formatter))

axs[0].set_ylabel("Concentration (microgram/L)")
plt.tight_layout()
plt.show()

# PIE CHARTS_ SEASONAL COMPOSITION
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

labels = ['Diatom', 'Dino', 'Hapto', 'Dictyo', 'Chloro']

def plot_pie(data, ax, title):
    sizes = data[phyto_cols].sum()
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', textprops={'size': 'small'})
    ax.set_title(title)

plot_pie(spring, axs[0], "Spring")
plot_pie(summer, axs[1], "Summer")
plot_pie(fall, axs[2], "Fall")

plt.tight_layout()
plt.show()