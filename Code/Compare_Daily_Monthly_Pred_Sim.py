# -*- coding: utf-8 -*-
"""
The following code is used to compare the scatter plots of obs. vs prediction
from two different sources:
    1-Conceptual simulation
    2-AI model (Meta Model)
The AI predictions are obtained from:
    USGS08353000_MetaLearner_v1
 Anlaysis are done at two timescales:
     1- Daily
     2- Monthly

The data associated with the gauge station can be found at:
    Source: https://waterdata.usgs.gov/nwis/inventory/?site_no=08353000
"""
#%%
#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
#%%
#plot settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
from matplotlib import rcParams
rcParams['font.family'] = 'Calibri'
#%%
#read data of interest
file_name='historical_simulation_cfs'
df_sim=pd.read_csv(file_name+'.csv').loc[:,['date','RioPuerco.Gage Inflow']]


#The first number indicates the number of neurons, and the second number is the learning rate
df_pred=pd.read_csv('Meta_5096_0.0010.csv')

#we filter the dates that overlap
df_pred = df_pred[df_pred['date'].isin(df_sim['date'])]
df_sim = df_sim[df_sim['date'].isin(df_pred['date'])]
#%%
# Aggregate to monthly flow
# Set date as index and resample by month
df_sim['date'] = pd.to_datetime(df_sim['date'])
df_monthly_sim = (
        df_sim.set_index('date')['RioPuerco.Gage Inflow']
        .resample('M')
        .sum()  # or mean()
        .reset_index()
    )

df_pred['date'] = pd.to_datetime(df_pred['date'])
df_monthly_pred = (
        df_pred.set_index('date')['med']
        .resample('M')
        .sum()  # or mean()
        .reset_index()
    )
df_monthly_obs= (
        df_pred.set_index('date')['obs']
        .resample('M')
        .sum()  # or mean()
        .reset_index()
    )
#%%
#plot scatter plots
# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)


# NSE calculation
def nse(obs, sim):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)



x1=df_pred['med']
y1=df_pred['obs']



x2=df_monthly_pred['med']
y2=df_monthly_obs['obs']


nse1 = nse(y1, x1)
nse2 = nse(y2, x2)
# Subplot 1
sc1 = axes[0].scatter(x1, y1, c=y1, cmap='viridis', edgecolor='k', s=50)
axes[0].plot([x1.min(), x1.max()], [x1.min(), x1.max()], 'r--', label='1:1 line')
axes[0].set_xlabel('Prediction (cfs)')
axes[0].set_ylabel('Observed (cfs)')
axes[0].set_title('Daily')
axes[0].legend()

# NSE box on top right
props = dict(boxstyle='round', facecolor='white', edgecolor='black', linestyle='dashed')
axes[0].text(0.95, 0.1, f'NSE = {nse1:.2f}', transform=axes[0].transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=props)

# Subplot 2
sc2 = axes[1].scatter(x2, y2, c=y2, cmap='plasma', edgecolor='k', s=50)
axes[1].plot([x2.min(), x2.max()], [x2.min(), x2.max()], 'r--', label='1:1 line')
axes[1].set_xlabel('Prediction (cfs)')
axes[1].set_ylabel('Observed (cfs)')
axes[1].set_title('Monthly')


# NSE box on top right
axes[1].text(0.95, 0.1, f'NSE = {nse2:.2f}', transform=axes[1].transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=props)

# Optional colorbars
fig.colorbar(sc1, ax=axes[0])
fig.colorbar(sc2, ax=axes[1])

plt.tight_layout()
plt.savefig('Compare_Scatter_Daily_Monthly.png')
#%%
# Example KGE function
def kge(sim, obs):
    sim = np.asarray(sim)
    obs = np.asarray(obs)

    cc = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    return 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)



# Add month as a new column
df_pred['month'] = df_pred['date'].dt.month

# Calculate KGE per month
monthly_kge = df_pred.groupby('month').apply(lambda g: kge(g['med'], g['obs']))

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
monthly_kge.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')

ax.set_xticks(range(12))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
ax.set_ylabel('KGE')
ax.set_title('Monthly KGE of Daily Predictions')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#%%
# Calculate KGE per month
monthly_obs = df_pred.groupby('month')['obs'].mean()

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
monthly_obs.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')

ax.set_xticks(range(12))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
ax.set_ylabel('Q (cfs)')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()