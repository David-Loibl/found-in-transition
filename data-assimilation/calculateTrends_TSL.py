#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:58:11 2018

@author: jnitzbon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import time

def linear(x, const, a):
	return const + a * x

#%% load all RGI and TSL data
    
df_TSL = pd.read_csv( '../data/TSL_full.csv' , parse_dates=[5], header=0)

df_RGI = pd.read_csv( '../data/RGI-Asia/rgi60_Asia.csv' , header=0, index_col=0)
df_RGI = df_RGI[ df_RGI.Area > 0.5 ]

#%% work with testset 
#nTest = 100
#np.random.seed(0)
#testset = [ df_RGI.index[i] for i in np.random.randint(low=0, high=len(df_RGI.index), size=nTest) ]

#df_RGI = df_RGI[ df_RGI.index.isin( testset ) ]

#%%

df_RGI_TSLtrends = pd.DataFrame( index=df_RGI.index )
nGlac = len(df_RGI)



#%% initialize dataframe for trends

obs = [ 'TSL_ELEV' ]

for o in obs:
    df_RGI_TSLtrends[str('TSLmax_trend')]=np.nan
    df_RGI_TSLtrends[str('TSLmax_r')]=np.nan
    
    df_RGI_TSLtrends[str('doyTSLmax_trend')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmax_r')]=np.nan
    
    df_RGI_TSLtrends[str('TSLmaxASO_trend')]=np.nan
    df_RGI_TSLtrends[str('TSLmaxASO_r')]=np.nan
    
    df_RGI_TSLtrends[str('doyTSLmaxASO_trend')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmaxASO_r')]=np.nan


#%% do everything for an example glacier first
#    rgi=df_RGI.index[4]

    
for rgi in df_RGI.index:

        
    tstart=time.time()
    
    print('At glacier ' + rgi + ' (' + str(df_RGI.index.get_loc(rgi)) + '/' + str(nGlac) + ')' )
    
    print('\t' + 'Loading TSL data ...')
    
    
    data = df_TSL[ df_TSL.RGIId == rgi ]
    data = data.set_index( 'LS_DATE' )
    
    #% calculate all trends and correlation coefficients
    for o in obs:
        print('\t' + 'Calculating trends in ' + o + ' ...')
        
        # max TSL
        ydata = data[o].resample('A').max()
        xdata = ydata.index.to_julian_date()
        trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
        
        df_RGI_TSLtrends.loc[ rgi, str('TSLmax_trend') ] = trend
        df_RGI_TSLtrends.loc[ rgi, str('TSLmax_r') ] = r
        
        # trend in timing of max TSL    
        ydata = data[o].groupby(pd.Grouper(freq='A')).idxmax().dt.dayofyear
        xdata = ydata.index.to_julian_date()
        trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)   
        
        df_RGI_TSLtrends.loc[ rgi, str('doyTSLmax_trend') ] = trend
        df_RGI_TSLtrends.loc[ rgi, str('doyTSLmax_r') ] = r
        
        # max TSL restricted to ASO
        ydata = data[o].resample('Q-OCT').max()
        ydata = ydata[ ydata.index.month == 10 ]
        xdata = ydata.index.to_julian_date()
        trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
        
        df_RGI_TSLtrends.loc[ rgi, str('TSLmaxASO_trend') ] = trend
        df_RGI_TSLtrends.loc[ rgi, str('TSLmaxASO_r') ] = r
        
        # trend in timing of max TSL  restricted to ASO
        ydata = data[o][ (data[o].index.month >= 8) & (data[o].index.month<=10)]
        ydata = data[o].resample('A').max()
        xdata = ydata.index.to_julian_date()
        trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)   
        
        df_RGI_TSLtrends.loc[ rgi, str('doyTSLmaxASO_trend') ] = trend
        df_RGI_TSLtrends.loc[ rgi, str('doyTSLmaxASO_r') ] = r
        
    tend=time.time()
    
    print('\t' + 'Done in ' + str(tend-tstart) + ' sec.')
    
#%% 
    
df_RGI = df_RGI.join(df_RGI_TSLtrends)

#%%

df_RGI.to_csv('../data/RGI+TSLtrends.csv')


