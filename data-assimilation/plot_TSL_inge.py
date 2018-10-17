#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:08:54 2018

@author: jnitzbon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% read in TSL data

#df_TSL = pd.read_csv( './data/TSL_full.csv' , parse_dates=[5], header=0)

df_TSL = pd.read_csv( '../data/TSL_full.csv' , parse_dates=[5], header=0, index_col=5)

df_RGI = pd.read_csv( '../data/RGI-Asia/rgi60_Asia.csv' , header=0)
df_RGI = df_RGI[ df_RGI.Area > 0.5 ]

#%% plot all TSL data

#fig, ax = plt.subplots()

#ax.plot_date(  df_TSL.index, df_TSL.TSL_ELEV, marker=',' )

#%%

#from scipy.optimize import curve_fit

def linear(x, const, a):
	return const + a * x

from scipy.stats import linregress

#%% trend analysis

#Nglacier=len(df_RGI) # 5min for 1000
Nglacier=10

i = 0
df_trend = df_RGI.iloc[:,0:1]
df_trend['Y_slope'] = np.zeros(len(df_RGI))
df_trend['Y_r'] = np.zeros(len(df_RGI))
df_trend['M01_slope'] = np.zeros(len(df_RGI))
df_trend['M01_r'] = np.zeros(len(df_RGI))
df_trend['M02_slope'] = np.zeros(len(df_RGI))
df_trend['M02_r'] = np.zeros(len(df_RGI))
df_trend['M03_slope'] = np.zeros(len(df_RGI))
df_trend['M03_r'] = np.zeros(len(df_RGI))
df_trend['M04_slope'] = np.zeros(len(df_RGI))
df_trend['M04_r'] = np.zeros(len(df_RGI))
df_trend['M05_slope'] = np.zeros(len(df_RGI))
df_trend['M05_r'] = np.zeros(len(df_RGI))
df_trend['M06_slope'] = np.zeros(len(df_RGI))
df_trend['M06_r'] = np.zeros(len(df_RGI))
df_trend['M07_slope'] = np.zeros(len(df_RGI))
df_trend['M07_r'] = np.zeros(len(df_RGI))
df_trend['M08_slope'] = np.zeros(len(df_RGI))
df_trend['M08_r'] = np.zeros(len(df_RGI))
df_trend['M09_slope'] = np.zeros(len(df_RGI))
df_trend['M09_r'] = np.zeros(len(df_RGI))
df_trend['M10_slope'] = np.zeros(len(df_RGI))
df_trend['M10_r'] = np.zeros(len(df_RGI))
df_trend['M11_slope'] = np.zeros(len(df_RGI))
df_trend['M11_r'] = np.zeros(len(df_RGI))
df_trend['M12_slope'] = np.zeros(len(df_RGI))
df_trend['M12_r'] = np.zeros(len(df_RGI))

df_trend.iloc[:,1:27] = np.nan
for RGI in df_RGI.RGIId[0:Nglacier]:
    # subset data
    tmp =  df_TSL.TSL_ELEV[df_TSL.RGIId == RGI ]
    
    #fig, ax = plt.subplots()
    
    #annual
    max_TSL = tmp.resample('AS').max()
    
    ydata = max_TSL
    xdata = ydata.index.to_julian_date()/365
    #[popt, pcov] = curve_fit( linear, xdata, ydata)
    [slope,offset,r,tmp2,tmp3] = linregress(xdata,ydata)
    df_trend.iloc[i,1] = slope
    df_trend.iloc[i,2] = r
    
    #ax.scatter( xdata, ydata , label="maximum TSL", color='r')
    #ax.plot( xdata, linear(xdata, offset, slope), label=('linear fit, trend=%0.4f m/year' % (slope)), color='r' )
    
    #monthly
    for j in range(1,13):
        ydata = tmp[tmp.index.month == j].resample('AS').mean()
        if len(ydata)>1:
            ydata = ydata[~np.isnan(ydata)]
            xdata = ydata.index.to_julian_date()/365
            [slope,offset,r,tmp2,tmp3] = linregress(xdata,ydata)
            df_trend.iloc[i,j*2+1] = slope
            df_trend.iloc[i,j*2+2] = r
        
        #ax.scatter( xdata, ydata , label="maximum TSL", color='g')
        #ax.plot( xdata, linear(xdata, offset, slope), label=('linear fit, trend=%0.4f m/year' % (slope)), color='r' )
        
    i = i+1
    
del tmp; del max_TSL; del ydata; del xdata; del slope; del offset
del r; del tmp2; del tmp3; del i; del j


#plt.hist(df_trend['Y_slope'][~np.isnan(df_trend['Y_slope'])],bins=40)

Median_trend = df_trend.median(axis=0, skipna=True,numeric_only=True)
  

#fig = plt.figure()
#for i in range(1,27,2):
#    j = np.ceil(i/2)
#    ax = fig.add_subplot(4,4,j)
#    ax.hist(df_trend.iloc[:,i][~np.isnan(df_trend.iloc[:,i])],bins=40,range=(-100, 100))
#    ax.set_title(df_trend.columns[i])





