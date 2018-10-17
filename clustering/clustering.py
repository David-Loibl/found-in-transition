# Clustering of TSL data for Himalayan glaciers
# Geo.X autumn school 2018

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def list_of_hex_colours(N, base_cmap):
    """
    Return a list of colors from a colourmap as hex codes

        Arguments:
            cmap: colormap instance, eg. cm.jet.
            N: number of colors.

        Author: FJC
    """
    cmap = cm.get_cmap(base_cmap, N)

    hex_codes = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        hex_codes.append(mcolors.rgb2hex(rgb))
    return hex_codes

def AverageEuclidianDifference(x, y):
    """
    Find the average Euclidian difference between two arrays x and y:
    d = sqrt(sum(x-y)^2)/n
    Liao, 2005,  doi:10.1016/j.patcog.2005.01.025
    """
    n = len(x)
    d = (np.sqrt(np.sum((x - y)*(x - y))))/n
    return d

# read in the csv file to a pandas dataframe
DataDirectory = '../data/'
df = pd.read_csv(DataDirectory+'TSL+RGI.csv',parse_dates=True, index_col='LS_DATE')
print(df)


# new dataframe with the regularly spaced time data
dr = pd.date_range(start='2013-03-31', end='2018-12-31',freq='M')

# get the IDs of each glacier
ids = df['RGIId'].unique()

# array to store the TSL data for clustering
ts = []
new_ids = []
# n_glaciers = 50 # how many you want to cluster

# get the data into the array and make a figure
plt.figure()

for i in range(len(ids)):
    reg_array = np.empty(len(dr))
    reg_array.fill(np.nan)
    # print(reg_array)
    # mask the dataframe to this id
    this_df = df[df['RGIId'] == ids[i]]
    # resample to monthly date
    monthly_tsl = this_df.TSL_ELEV.resample('M').mean()
    monthly_tsl = monthly_tsl.interpolate(method='linear')
    if(monthly_tsl.index[0] == dr[0]):
        plt.plot(monthly_tsl)
        ts.append(monthly_tsl)
        new_ids.append(ids[i])

print(ts)

plt.savefig(DataDirectory+'tsl_monthly.png', dpi=300)
# # plt.show()
plt.clf()

print("Starting the clustering...")

n = len(ts)
# now do the clustering
# correlation coefficients
cc = np.zeros(int(n * (n - 1) / 2))
#cc = []
k = 0
for i in range(n):
    for j in range(i+1, n):
        tsi = ts[i]
        tsj = ts[j]
        if len(tsi) > len(tsj):
            tsi = tsi[:len(tsj)]
        else:
            tsj = tsj[:len(tsi)]
        #cc[k] = np.corrcoef(tsi, tsj)[0, 1]
        cc[k] = AverageEuclidianDifference(tsi,tsj)
        k += 1

# take arccos of the correlation coefficients
#d = np.arccos(cc)
# linkage matrix
ln = linkage(cc, method='complete')
# define the threshold for cutoff = kind of determines the number of clusters
thr = 60

# compute cluster indices
cl = fcluster(ln, thr, criterion = 'distance')
print("I've finished! I found {} clusters for you :)".format(cl.max()))
print([int(c) for c in cl])

# set colour palette: 8 class Set 1 from http://colorbrewer2.org
N_colors = 8
colors = list_of_hex_colours(N_colors, 'Set1')[:cl.max()]
set_link_color_palette(colors)

#plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('distance', fontsize=14)
R = dendrogram(ln, color_threshold=thr+0.5, above_threshold_color='k',no_labels=True)

plt.axhline(y = thr, color = 'r', ls = '--')
plt.savefig(DataDirectory+'tsl_dendrogram.png', dpi=300)
plt.clf()

# make plots of the profile individually for each cluster
# assign the cluster id to the dataframe
for i in range(len(new_ids)):
    df.loc[df.RGIId == new_ids[i], 'cluster_id'] = cl[i]

fig, ax = plt.subplots(nrows=1, ncols=cl.max(), figsize=(10,5))
# make a big subplot to allow sharing of axis labels
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#ax = ax.ravel()

for i, c in enumerate(cl):
    ax[c-1].plot(ts[i], c=colors[c-1])

for a in range(len(ax)):
    ax[a].set_ylim(4000,7000)

# plt.show()
plt.xlabel('Year')
plt.ylabel('Transient snow line altitude (m a.s.l.)')
plt.savefig(DataDirectory+'subplots_clustered.png', dpi=300)
