# -*- coding: utf-8 -*-
"""
Plot a heatmap of R² between streamflow columns in the DataFrame.
Heatmap rows and columns are labeled with numbers, and a legend maps them to column names.
While general, this code is specifically deployed for:
USGS 08353000 RIO PUERCO NEAR BERNARDO, NM

Catchment properties:
    
    
DESCRIPTION:
Latitude 34°24'37",   Longitude 106°51'16"   NAD83
Socorro County, New Mexico, Hydrologic Unit 13020204
Drainage area: 6,437 square miles
Contributing drainage area: 5,307 square miles,
Datum of gage: 4,721.92 feet above NAVD88.
Source: https://waterdata.usgs.gov/nwis/inventory/?site_no=08353000
"""
#%%
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
#%%
# Drop 'date' and extract data columns
os.chdir(r'TOVA-tech-interview\data') #change this
file_list=[r'historical_simulation_cfs']
filename=file_list[0]
df=pd.read_csv(filename+'.csv')
df_data = df.drop(columns=['date'], errors='ignore')
col_names = df_data.columns.tolist()
#%%
# Compute R² correlation matrix
r = df_data.corr() ** 2

# Create numeric labels
labels = [name.split('.')[0] for name in col_names]
unique_labels = sorted(set(labels))
label_to_number = {label: i+1 for i, label in enumerate(unique_labels)}
numeric_labels = [label_to_number[label] for label in labels]

# Reverse map for legend
number_to_label = {v: k for k, v in label_to_number.items()}
legend_entries = [f"{num}: {name}" for num, name in number_to_label.items()]

# Arrange legend entries into columns
n = len(legend_entries)
legend_cols=5
rows = math.ceil(n / legend_cols)
legend_lines = []

for i in range(rows):
    line_entries = []
    for j in range(legend_cols):
        idx = i * legend_cols + j
        if idx < n:
            line_entries.append(f"{legend_entries[idx]:<20}")  # padded for spacing
    legend_lines.append("  ".join(line_entries))

legend_text = "\n".join(legend_lines)

# Plot heatmap
fig, ax = plt.subplots(figsize=(15, 15),dpi=300)
im = ax.imshow(r, cmap='viridis', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(numeric_labels)))
ax.set_yticks(np.arange(len(numeric_labels)))
ax.set_xticklabels(numeric_labels, fontsize=15)
ax.set_yticklabels(numeric_labels, fontsize=15)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add R² values inside heatmap cells
for i in range(len(r)):
    for j in range(len(r)):
        val = r.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                color="white" if val < 0.5 else "black")

ax.set_title("R² Heatmap of Streamflow Stations")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("R²", fontsize=15)

# Add multiline legend using figtext
plt.figtext(0.5, 0.05, legend_text, ha="center", va="top", fontsize=15, family="monospace")

plt.tight_layout()
plt.show()