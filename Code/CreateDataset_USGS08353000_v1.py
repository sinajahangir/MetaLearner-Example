# -*- coding: utf-8 -*-
"""
This code is used to create train and test datasets for DL (e.g., LSTM)
model devlopment

The data is comprised of 27 static catchments from Kratzert et al (2019) and 3
dynamic varibales from Daymet (v4?; Thornton et al., 2022): pr, tmax, and tmin

Reference:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., 2019. Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences 23, 5089–5110. https://doi.org/10.5194/hess-23-5089-2019

Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022.
Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129

"""
#%%
#import necessary libraries
import numpy as np
import pandas as pd
import os
#%%
#We need to read the refernce file to retrive the static catchment features
csv_path='All421data_Test_st.csv'
df = pd.read_csv(csv_path)
#%%
#static features
col_list=df.columns.to_list()[-27:]
#%%
#read climatology data for the location of interest
gauge='RioPuerco'
pr=pd.read_csv('historical_observations_prcp.csv').loc[:,gauge+'.Gage Inflow']
tmax=pd.read_csv('historical_observations_tmax.csv').loc[:,gauge+'.Gage Inflow']
tmin=pd.read_csv('historical_observations_tmin.csv').loc[:,gauge+'.Gage Inflow']
q=pd.read_csv('historical_observations_cfs.csv').loc[:,gauge+'.Gage Inflow']
date=pd.read_csv('historical_observations_cfs.csv').loc[:,'date']
#%%
#populate the new df
df_new=pd.DataFrame({
    'date': date,
    'pr': pr,
    'tmax': tmax,
    'tmin': tmin,
    'q': q
})
#this is the closest catchment to the gauge stations
#it is noted that as we are fine-tuning, exact values are not important
basin_id=343
for ii, col in enumerate(col_list):
    df_new[col]=df[df['basin_id']==basin_id].loc[:,col].iloc[0]
#%%
#split to training/validation and testing
df_train=df_new.iloc[:int(0.85*len(df_new))]
df_test=df_new.iloc[int(0.85*len(df_new)):]
#%%
#save to csv
df_train.to_csv('Train_USGS08353000_v1.csv',index=None)
df_test.to_csv('Test_USGS08353000_v1.csv',index=None)

