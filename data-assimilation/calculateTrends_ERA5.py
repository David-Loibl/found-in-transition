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

#%%

df_RGI = pd.read_csv( '../data/RGI-Asia/rgi60_Asia.csv' , header=0, index_col=0)
df_RGI = df_RGI[ df_RGI.Area > 0.5 ]
nGlac = len(df_RGI)



#%% work with testset 

test=False

if test:
    nTest = 200
    np.random.seed(0)
    testset = [ df_RGI.index[i] for i in np.random.randint(low=0, high=len(df_RGI.index), size=nTest) ]
    df_RGI = df_RGI[ df_RGI.index.isin( testset ) ]
    nGlac = len(df_RGI)

#%% initialize dataframe for trends


df_RGI_ERA5trends = pd.DataFrame( index=df_RGI.index )





obs = [ 'tmean', 'tp', 'Gmean', 'wsmean', 'CCmean' ]
samp = [ 'Annual', 'JFM', 'AMJ', 'JAS', 'OND' ] + [ 'Monthly' + str(m) for m in np.arange(1,13) ]

for o in obs:
    for s in samp:
        df_RGI_ERA5trends[str(o+s+'_trend')]=np.nan
        df_RGI_ERA5trends[str(o+s+'_r')]=np.nan
        df_RGI_ERA5trends[str(o+s+'_p')]=np.nan


#%% do everything for an example glacier first

   # rgi='RGI60-15.00205'

for rgi in df_RGI.index:
    
    tstart=time.time()
    
    print('At glacier ' + rgi + ' (' + str(df_RGI.index.get_loc(rgi)) + '/' + str(nGlac) + ')' )
    
    print('\t' + 'Loading ERA5 data ...')


    with open( str( '../data/era5/' + rgi + '.pkl' ), 'rb') as f:    
        data = pickle.load(f)   

#%%
    
    # calculate all trends and correlation coefficients
    seasonmap = {'JFM':3,'AMJ':6,'JAS':9,'OND':12}
    for o in obs:
        
        print('\t' + 'Calculating trends in ' + o + ' ...')
        
        # observables which are averaged over sampling period
        if o in ['tmean', 'Gmean', 'wsmean', 'CCmean' ]:
        
            #annual
            s='Annual'
            
            ydata = data[o].resample('A').mean()
            xdata = ydata.index.to_julian_date()    
            trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
            
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_trend') ] = trend
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_r') ] = r
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_p') ] = p
            
            #seasonal
            for s in seasonmap.keys():
                ydata = data[o].resample('Q-DEC').mean()
                ydata = ydata[ ydata.index.month==seasonmap[s] ]
                xdata = ydata.index.to_julian_date()    
                trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
                
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_trend') ] = trend
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_r') ] = r
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_p') ] = p
                
            #monthly
            for m in np.arange(1,13):
                ydata = data[o][ data[o].index.month==m ]
                xdata = ydata.index.to_julian_date()    
                trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)

                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_trend') ] = trend
                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_r') ] = r
                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_p') ] = p                

                
        elif o in ['tp']:
            
            #annual
            s='Annual'
            
            ydata = data[o].resample('A').sum()
            xdata = ydata.index.to_julian_date()    
            trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
            
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_trend') ] = trend
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_r') ] = r
            df_RGI_ERA5trends.loc[ rgi, str(o+s+'_p') ] = p
            
            #seasonal
            for s in seasonmap.keys():
                ydata = data[o].resample('Q-DEC').sum()
                ydata = ydata[ ydata.index.month==seasonmap[s] ]
                xdata = ydata.index.to_julian_date()    
                trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)
                
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_trend') ] = trend
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_r') ] = r
                df_RGI_ERA5trends.loc[ rgi, str(o+s+'_p') ] = p
                
            #monthly
            for m in np.arange(1,13):
                ydata = data[o][ data[o].index.month==m ]
                xdata = ydata.index.to_julian_date()    
                trend, offset, r, p, trend_unc = stats.linregress(xdata, ydata)

                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_trend') ] = trend
                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_r') ] = r
                df_RGI_ERA5trends.loc[ rgi, str(o+'Monthly'+str(m)+'_p') ] = p       
            
        else:
            print('Invalid observable...')
            
    tend=time.time()
    
    print('\t' + 'Done in ' + str(tend-tstart) + ' sec.')
    
#%% 

if test:
    df_RGI_ERA5trends.to_csv('../data/ERA5trends_testset.csv')
else:
    df_RGI_ERA5trends.to_csv('../data/ERA5trends_full.csv')


