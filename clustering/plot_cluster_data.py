import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Plotting scripts for the csv with the cluster data
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

def MakeBoxPlotByCluster():
    """
    Make a boxplot showing the stats for each cluster
    """

    # read the csv and get some info
    df = pd.read_csv(DataDirectory+'TSL+RGI_clustered.csv')
    df = df[np.isnan(df['cluster_id']) == False]
    df = df.sort_values(by='cluster_id')

    print("========SOME CLUSTER STATISTICS=========")
    clusters = df['cluster_id'].unique()
    for cl in clusters:
        this_df = df[df['cluster_id'] == cl]
        print("Cluster {}, median slope = {}".format(cl, this_df['Slope'].median()))
        print("Cluster {}, median aspect = {}".format(cl, this_df['Aspect'].median()))
        print("Cluster {}, median elevation = {}".format(cl, this_df['Zmed'].median()))
        print("Cluster {}, median TSL elev = {}".format(cl, this_df['TSL_ELEV'].median()))
    print("========================================")

    # set props for fliers
    flierprops = dict(marker='o', markerfacecolor='none', markersize=1,
                  linestyle='none', markeredgecolor='k')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
    axes = axes.ravel()
    # make a big subplot to allow sharing of axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    col_keys = ['Slope', 'Aspect', 'Zmed', 'TSL_ELEV']
    labels = ['Slope', 'Aspect', 'Median elevation (m)', 'Transient snow line (m a.s.l.)']

    N_colors = 8
    colors = list_of_hex_colours(N_colors, 'Set1')
    for i, this_ax in enumerate(axes):
        this_ax.set_ylabel(labels[i])
        this_ax.set_xlabel('')
        # make the boxplot and return the dict with the boxplot properties
        bp_dict = df.boxplot(column=col_keys[i], by=['cluster_id'], return_type='both', patch_artist=True, flierprops=flierprops, ax=this_ax)
        # make the median lines black
        #[[item.set_color('k') for item in bp_dict[key]['medians']] for key in bp_dict.keys()]

        # change the colours based on the cluster ID
        for row_key, (ax,row) in bp_dict.iteritems():
            this_ax.set_xlabel('')
            j=-1 #stupid thing because there are double the number of caps and whiskers compared to boxes
            for i,cp in enumerate(row['caps']):
                if i%2==0:
                    j+=1
                # cp.set(color=colors[j])
                cp.set(color='k')
            j=-1
            for i,wh in enumerate(row['whiskers']):
                if i%2==0:
                    j+=1
                # wh.set_color(colors[j])
                wh.set_color('k')
            for i,box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
                box.set_alpha(0.7)
                # box.set_edgecolor(colors[i])
                box.set_edgecolor('k')
            for i,med in enumerate(row['medians']):
                med.set(color='k')
            for i,pt in enumerate(row['fliers']):
                pt.set_markeredgecolor(colors[i])


        ax.grid(color='0.8', linestyle='--', which='major', zorder=1)
        plt.subplots_adjust(wspace=0.3,left=0.25,right=0.9, hspace=0.35, bottom=0.1)
        x_labels = [str((int(x))) for x in df.cluster_id.unique()]
        ax.set_xticklabels(x_labels, fontsize=12)
        #print(boxplot)
    plt.suptitle('')
    plt.xlabel('Cluster number')
    plt.tight_layout()

        # ax.set_ylabel('Catchment relief (m)', fontsize=14)

    #plt.subplots_adjust(left=0.2)
    plt.savefig(DataDirectory+'cluster_boxplot.png', dpi=300)
    plt.clf()

DataDirectory = '../data/'
MakeBoxPlotByCluster()
