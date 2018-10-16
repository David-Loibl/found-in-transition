# Clustering of TSL data for Himalayan glaciers
# Geo.X autumn school 2018

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

# read in the csv file to a pandas dataframe
DataDirectory = '/home/fiona/geox/challenge/'
df = pd.read_csv(DataDirectory+'TSL+RGI.csv',parse_dates=True, index_col='LS_DATE')
print(df)

# new dataframe with the regularly spaced time data
dr = pd.date_range(start='2013-01-01', end='2018-12-31',freq='M')
rs = pd.DataFrame(index=dr)
print(rs)

# get the IDs of each glacier
ids = df['RGIId'].unique()

# set up 2d array for clustering. We need an array of n_glaciers (len(ids)) times 6 years * 12 months (72)
n_years = 6
ts = np.empty((len(ids), n_years*12))

plt.figure()

for i, id in enumerate(ids):
    # mask the dataframe to this id
    this_df = df[df['RGIId'] == id]
    # resample to monthly date
    #monthly_tsl = this_df.TSL_ELEV.resample('M').mean()
    plt.plot(this_df.TSL_ELEV.resample('M').mean())
    #ts[i] = this_df.TSL_ELEV.resample('M').mean().index[0]

plt.show()
