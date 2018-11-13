#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:20:13 2018

@author: jnitzbon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time

def linear(x, const, a):
	return const + a * x

# load RGI data from testset
df = pd.read_csv( '../data/RGI-Asia/rgi60_Asia.csv' , header=0, index_col=0)
df = df[ df.Area > 0.5 ]

#%% 
#nTest = 200
#np.random.seed(0)
#testset = [ df.index[i] for i in np.random.randint(low=0, high=len(df.index), size=nTest) ]
#df = df[ df.index.isin( testset ) ]

#%% add trends of TSL and ERA5
df = df.join(  pd.read_csv( '../data/ERA5trends_full.csv', index_col=0 ) )
df = df.join(  pd.read_csv( '../data/TSLtrends_full.csv', index_col=0 ) )

#%% statistics on significant trends
trendObservables = df.filter(like='trend').columns
trendObservables = list(map(lambda x: x.replace('_trend', ''), trendObservables))

for obs in trendObservables:
    sigFrac = np.sum( df[ str(obs+'_p') ] < 0.05 ) / df[ str(obs+'_p') ].count() 
    print('Fraction of significant (p<0.05) trends in ' + obs + ': ' + str(sigFrac) )
    fig, ax = plt.subplots()
    sigTrends = df[ str(obs+'_trend') ][ df[ str(obs+'_p') ] < 0.05 ]*365
    ax.hist( sigTrends, label='data' )
    ax.set_title( 'Histogram (n=' + str( np.sum( df[ str(obs+'_p') ] < 0.05 ) ) + ') of significant (p<0.05) trends in ' + obs  ) 
    ax.set_xlabel( obs + ' trend [unit-of-obs / year]')
    ax.axvline( sigTrends.median(), 0, 1, label='median', color='k', linestyle=':' )
    ax.axvline( sigTrends.mean(), 0, 1, label='mean', color='r', linestyle='--' )
    try:
        ax.set_xlim( [ -np.max(np.abs(sigTrends)), +np.max(np.abs(sigTrends)) ] )
    except:
        pass
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig( './plots/histogram_' + obs + '_significant.png' )

#%% correlation plots

# scatter plots topological
topologicObservables=[ 'CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax']
obs1 = 'TSLmaxASO'
for obs2 in topologicObservables:  
    obss = [ str(obs1+'_trend'), obs2 ]
    mask = df[ str(obs1+'_p') ] < 0.05
    pd.plotting.scatter_matrix( df[ obss ][ mask ] , diagonal='kde', alpha=0.5)

#%% TSL max versus trends
trendObservables = df.filter(like='trend').columns
trendObservables = list(map(lambda x: x.replace('_trend', ''), trendObservables))    
obs1 = 'TSLmaxASO' 
for obs2 in trendObservables:
    obss = [ str(obs1+'_trend'), str(obs2+'_trend') ]
    mask = (df[ str(obs1+'_p') ] < 0.05) & (df[ str(obs2+'_p') ] < 0.05)
    pd.plotting.scatter_matrix( df[ obss ][ mask ]*365 , diagonal='kde', alpha=0.5)
