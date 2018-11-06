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
# exclude 2018 data for trend as no full season
df_TSL = df_TSL[ df_TSL['LS_DATE'].dt.year < 2018 ]

df_RGI = pd.read_csv( '../data/RGI-Asia/rgi60_Asia.csv' , header=0, index_col=0)
df_RGI = df_RGI[ df_RGI.Area > 0.5 ]

#%% work with testset 
#nTest = 200
#np.random.seed(0)
#testset = [ df_RGI.index[i] for i in np.random.randint(low=0, high=len(df_RGI.index), size=nTest) ]
#df_RGI = df_RGI[ df_RGI.index.isin( testset ) ]

#%%



df_RGI_TSLtrends = pd.DataFrame( index=df_RGI.index )
nGlac = len(df_RGI)



# initialize dataframe for trends

obs = [ 'TSL_ELEV' ]

for o in obs:
    df_RGI_TSLtrends[str('TSLmax_trend')]=np.nan
    df_RGI_TSLtrends[str('TSLmax_r')]=np.nan
    df_RGI_TSLtrends[str('TSLmax_p')]=np.nan
    
    df_RGI_TSLtrends[str('doyTSLmax_trend')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmax_r')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmax_p')]=np.nan
    
    df_RGI_TSLtrends[str('TSLmaxASO_trend')]=np.nan
    df_RGI_TSLtrends[str('TSLmaxASO_r')]=np.nan
    df_RGI_TSLtrends[str('TSLmaxASO_p')]=np.nan
    
    df_RGI_TSLtrends[str('doyTSLmaxASO_trend')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmaxASO_r')]=np.nan
    df_RGI_TSLtrends[str('doyTSLmaxASO_p')]=np.nan


#%% do everything for an example glacier first

    #rgi='RGI60-15.00205'

for rgi in  df_RGI.index:

        
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
        mask = np.isfinite(ydata)
        if ydata.count()>2 : # at least three data points for linear regression
            trend, offset, r, p, trend_unc = stats.linregress(xdata[mask], ydata[mask])
            df_RGI_TSLtrends.loc[ rgi, str('TSLmax_trend') ] = trend
            df_RGI_TSLtrends.loc[ rgi, str('TSLmax_r') ] = r
            df_RGI_TSLtrends.loc[ rgi, str('TSLmax_p') ] = p
        
        # trend in timing of max TSL    
        ydata = data[o].resample('A').agg( lambda x : np.nan if x.count()== 0 else x.idxmax() ).dt.dayofyear
        xdata = ydata.index.to_julian_date()
        mask = np.isfinite(ydata)
        if ydata.count()>2 : # at least three data points for linear regression
            trend, offset, r, p, trend_unc = stats.linregress(xdata[mask], ydata[mask])
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmax_trend') ] = trend
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmax_r') ] = r
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmax_p') ] = p
        
        # max TSL restricted to ASO
        ydata = data[o][ (data[o].index.month >= 8) & (data[o].index.month<=10)]
        if ydata.count()>2 : # at least three data points for linear regression

            ydata = ydata.resample('A').max()
            xdata = ydata.index.to_julian_date()
            mask = np.isfinite(ydata)
            trend, offset, r, p, trend_unc = stats.linregress(xdata[mask], ydata[mask])
            df_RGI_TSLtrends.loc[ rgi, str('TSLmaxASO_trend') ] = trend
            df_RGI_TSLtrends.loc[ rgi, str('TSLmaxASO_r') ] = r
            df_RGI_TSLtrends.loc[ rgi, str('TSLmaxASO_p') ] = p

        # trend in timing of max TSL  restricted to ASO months
        ydata = data[o][ (data[o].index.month >= 8) & (data[o].index.month<=10) ]
        if ydata.count()>2 : # at least three data points for linear regression

            ydata = ydata.resample('A').agg( lambda x : np.nan if x.count() == 0 else x.idxmax() ).dt.dayofyear
            xdata = ydata.index.to_julian_date()
            mask = np.isfinite(ydata)
            trend, offset, r, p, trend_unc = stats.linregress(xdata[mask], ydata[mask])    
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmaxASO_trend') ] = trend
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmaxASO_r') ] = r
            df_RGI_TSLtrends.loc[ rgi, str('doyTSLmaxASO_p') ] = p
            
    tend=time.time()
    
    print('\t' + 'Done in ' + str(tend-tstart) + ' sec.')
        
    
#%%    
    
df_RGI_TSLtrends.to_csv('../data/TSLtrends_full.csv')

    

#%%
#df_RGI = df_RGI.join(df_RGI_TSLtrends)
#df_RGI.to_csv('../data/RGI+TSLtrends.csv')

