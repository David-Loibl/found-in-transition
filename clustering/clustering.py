#Clustering of TSL data for Himalayan glaciers
# Geo.X autumn school 2018

# import modules
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from glob import glob
import os

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

def DistanceVsNClusters(ln, threshold_level=0):
    """
    find the distance between each cluster compared to the
    number of clusters. Use this to determine where to put the distance
    threshold

    Args:
        ln: linkage matrix from clustering
        threshold_level: which level to return the threshold at. 0 = max distance between clusters.

    Author: FJC
    """

    #print ln
    # each iteration merges one cluster. so we start with n_clusters = n samples,
    # and then it reduces by one each time.
    clusters = []
    n_clusters = len(ln)+1
    for l in ln:
        # the distance is the 3rd column.
        this_dist = l[2]
        clusters.append(n_clusters)
        n_clusters -= 1

    # find the difference in the distances between each point in the linkage array
    dist = ln[:,2]
    #deltas = [j-i for i, j in zip(dist[:-1], dist[1:])]
    # get the argmax of the difference
    #i = np.argmax(deltas)
    n_clusters = clusters[threshold_level]
    # now find the distance threshold corresponding to this
    thr = dist[threshold_level]
    print ('The optimum distance threshold is '+str(thr))
   
    return thr

def Clustering(DataDirectory, ts):
    """
    Clustering of the time series
    ts = pandas groupby of the time series
    """

    print("Starting the clustering...")

    n = len(ts)
    # now do the clustering
    # correlation coefficients
    cc = np.zeros(int(n * (n - 1) / 2))
    #cc = []
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            tsi = np.asarray(ts.iloc[i])
            tsj = np.asarray(ts.iloc[j])
            #mask for nans
            bad = ~np.logical_or(np.isnan(tsi), np.isnan(tsj))
    #        print(bad)
            tsi = np.compress(bad, tsi)
            tsj = np.compress(bad, tsj)       
            #print(tsi,tsj) 

            if len(tsi) > len(tsj):
                tsi = tsi[:len(tsj)]
            else:
                tsj = tsj[:len(tsi)]
            #cc[k] = np.corrcoef(tsi, tsj)[0, 1]
            cc[k] = AverageEuclidianDifference(tsi,tsj)
            k += 1

    # get distance metric from the correlation coefficients
    #d = np.sqrt(2*(1 - cc))
    # print(cc)
    # linkage matrix
    #ln = linkage(d, method='complete')
    ln = linkage(cc, method='complete')
    # define the threshold for cutoff = kind of determines the number of clusters
    #thr = 175
    #thr = DistanceVsNClusters(ln, threshold_level)
    thr = 1.76

    # compute cluster indices
    cl = fcluster(ln, thr, criterion = 'distance')
    print("I've finished! I found {} clusters for you :)".format(cl.max()))
    print([int(c) for c in cl])

    # set colour palette: 8 class Set 1 from http://colorbrewer2.org
    N_colors = cl.max()
    colors = list_of_hex_colours(N_colors, 'Set2')
    set_link_color_palette(colors)

    #plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('distance', fontsize=14)
    R = dendrogram(ln, color_threshold=thr, above_threshold_color='k',no_labels=True)

    plt.axhline(y = thr, color = 'r', ls = '--')
    plt.savefig(DataDirectory+'tsl_dendrogram.png', dpi=300, transparent=True)
    plt.clf()

    # make plots of the profile individually for each cluster
    # assign the cluster id to the dataframe
    for i in range(len(ids)):
        full_df.loc[full_df.RGIId == ids[i], 'cluster_id'] = cl[i]

    full_df.to_csv(DataDirectory+'TSL+RGI_clustered.csv', index=False)

    fig, ax = plt.subplots(nrows=1, ncols=cl.max(), figsize=(12,5))
    # make a big subplot to allow sharing of axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #ax = ax.ravel()

    for i, c in enumerate(cl):
        ax[c-1].plot(ts[i], c=colors[c-1], lw=0.5)

    for a in range(len(ax)):
        ax[a].set_ylim(3000,7000)

    # plt.show()
    plt.xlabel('Month from March 2013', fontsize=15)
    plt.ylabel('Transient snow line altitude (m a.s.l.)', labelpad=20, fontsize=15)
    plt.tight_layout()
    plt.savefig(DataDirectory+'subplots_clustered.png', dpi=300, transparent=True)

# read in the csv file to a pandas dataframe
DataDirectory = '../data/'
df = pd.DataFrame()
for fname in glob(DataDirectory+'*interp-*.csv'):
#for fname in glob(DataDirectory+'*interp-0-1000.csv'):
    print(fname)
    this_df = pd.read_csv(fname, index_col='LS_DATE ')
    # print('Number of IDS: ', len(this_df['RGIId'].unique()))
    df = df.append(this_df)

print(df)
ids = df.RGIId.unique()
print("N GLACIERS: ", len(ids))

# # read in the full DataFrame
full_df = pd.read_csv(DataDirectory+'TSL+RGI.csv')
full_df = full_df[full_df['RGIId'].isin(ids)]
# full_df.to_csv(DataDirectory+'TSL+RGI_interp_only.csv', index=False)

# get the data into a list for clustering
ts = df.groupby('RGIId')['TSL_ELEV'].apply(list)
print(ts)

Clustering(DataDirectory, ts)

