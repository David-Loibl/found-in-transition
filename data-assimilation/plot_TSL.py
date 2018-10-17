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

df_TSL = pd.read_csv( './data/TSL_full.csv' , parse_dates=[5], header=0, index_col=5)

df_RGI = pd.read_csv( './data/RGI-Asia/rgi60_Asia.csv' , header=0)
df_RGI = df_RGI[ df_RGI.Area > 0.5 ]

#%% plot all TSL data

fig, ax = plt.subplots()

ax.plot_date(  df_TSL.LS_DATE, df_TSL.TSL_ELEV, marker=',' )

#%%

from scipy.optimize import curve_fit

def linear(x, const, a):
	return const + a * x


#%% trend analysis

Nglacier=5



for RGI in df_RGI.RGIId[0:Nglacier]:
    
    fig, ax = plt.subplots()
    
    #annual
    max_TSL = df_TSL.TSL_ELEV[df_TSL.RGIId == RGI ].resample('AS').max()
    
    ydata = max_TSL
    xdata = ydata.index.to_julian_date()
    [popt, pcov] = curve_fit( linear, xdata, ydata)
    ax.scatter( xdata, ydata , label="maximum TSL", color='r')
    ax.plot( xdata, linear(xdata, popt[0], popt[1]), label=('linear fit, trend=%0.4f m/year' % (popt[1]*365)), color='r' )
    
        
    med_TSL = df_TSL.TSL_ELEV[df_TSL.RGIId == RGI ].resample('AS').median()
    ydata = med_TSL
    xdata = ydata.index.to_julian_date()
    [popt, pcov] = curve_fit( linear, xdata, ydata)
    

    ax.scatter( xdata, ydata , label="median TSL", color='k')
    ax.plot( xdata, linear(xdata, popt[0], popt[1]), label=('linear fit, trend=%0.4f m/year' % (popt[1]*365)), color='k' )
    ax.legend()
        
    
    



  









