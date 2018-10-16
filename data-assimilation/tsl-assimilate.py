'''
Data assimilation routines for Transient Snowline Altitude (TSL) files,
in combintation with Randolph Glacier Inventory (RGI) v6, and ERA5 climate
timesseries data.

TODO
# Handle multiple measurements for the same glacier and timestep -> median (?)


'''

import pandas as pd
from os import path, walk

RGI_shapefile = 'data/RGI-Asia/rgi60_Asia.csv'
df_rgi = pd.read_csv(RGI_shapefile)

TSL_dir = 'data/gee-results/'

if path.isfile(TSL_dir + 'TSL_full.csv'):
    print('Found TSL_full.csv. Reading ...')
    df_full = pd.read_csv(TSL_dir + 'TSL_full.csv', header=0)
else:
    print('No TSL_full.csv found. Generating from files in raw/ dirctory ...')
    TSL_files = []
    for (dirpath, dirnames, filesnames) in walk(TSL_dir + 'raw'):
        TSL_files.extend(filesnames)
        break

    print(TSL_files)
    print(TSL_files[0])

    for i in range(len(TSL_files)):

        if i == 0:
            df_full = pd.read_csv(TSL_dir + 'raw/' + TSL_files[i], header=0)
        else:
            df = pd.read_csv(TSL_dir + 'raw/' + TSL_files[i], header=0)
            df_full = pd.concat([df_full, df], sort=True)

    df_full.drop('system:index', axis=1, inplace=True)
    df_full.drop('.geo', axis=1, inplace=True)

    # df = pd.read_csv('data/gee-results/TSL-results-103-104.csv', header=0)
    df_full['LS_DATE'] = pd.to_datetime(df_full.LS_SCENE.str[12:20])

if path.isfile(TSL_dir + 'TSL+RGI.csv'):
    print('Found TSL+RGI.csv. Reading ...')
    df_tsl_rgi = pd.read_csv(TSL_dir + 'TSL+RGI.csv', header=0)
else:
    df_tsl_rgi = pd.merge(df_full, df_rgi, on='RGIId', copy=False)

print(df_tsl_rgi.RGIId.unique())
for glacier_id in df_tsl_rgi.RGIId.unique():
    df_duplicate_measurements = df_tsl_rgi['LS_DATE']

df_chrono = df_tsl_rgi.sort_values(by=['LS_DATE'])
print(df_chrono['LS_DATE'])

# print(df['LS_SCENE'])
# print(df['LS_DATE'])
# print(df_full)
# print(df_full.shape)
# print(df_full.size)
# print(df_tsl_rgi.columns.values)
# print('Unique glaciers: %d' % len(df_full.RGIId.unique()))
# print(df_tsl_rgi)

# Write merged DataFrames to csv files
# df_full.to_csv(TSL_dir + 'TSL_full.csv', sep=',')
# df_tsl_rgi.to_csv(TSL_dir + 'TSL+RGI.csv', sep=',')




# plot.pyplot.scatter(df['LS_DATE'], df['TSL_ELEV'])
# plot.show()

#df.plot(x='LS_DATE', y='TSL_ELEV', style='k.')
#pyplot.show()


# data = [go.Scatter(x=df.LS_DATE, y=df.TSL_ELEV)]
# py.iplot(data)
