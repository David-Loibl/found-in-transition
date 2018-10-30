import sys
import pandas as pd
import numpy as np

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

print(sys.argv[1])

data_path = str(sys.argv[1])
start_glacier = int(sys.argv[2])
end_glacier = int(sys.argv[3])

# read in the csv file to a pandas dataframe
# DataDirectory = '../data/'
df = pd.read_csv(data_path+'TSL+RGI.csv',parse_dates=True, index_col='LS_DATE')
# print(df)


# new dataframe with the regularly spaced time data
dr = pd.date_range(start='2013-03-31', end='2018-12-31',freq='M')

# get the IDs of each glacier
ids = df['RGIId'].unique()

# array to store the TSL data for clustering
# ts = []
new_ids = []
n_glaciers = 50 # how many you want to cluster

# get the data into the array and make a figure
# plt.figure()
df_interpolate = pd.DataFrame() # columns=['RGIId', 'Month', 'TSL_ELEV']
new_df = pd.DataFrame()

for i in range(start_glacier, end_glacier):
    # print(reg_array)
    # mask the dataframe to this id
    this_df = df[df['RGIId'] == ids[i]]
    # resample to monthly date
    monthly_tsl = this_df.TSL_ELEV.resample('M').mean()
    monthly_tsl = monthly_tsl.interpolate(method='linear')
    if(monthly_tsl.index[0] == dr[0]):
        new_df  = monthly_tsl.to_frame()
        new_df['RGIId'] = ids[i]
        df_interpolate = df_interpolate.append(new_df)
        # plt.plot(monthly_tsl)
        #monthly_ds = {'RGIId': ids[i], 'Month': monthly_tsl.index, 'TSL_ELEV': monthly_tsl.values}
        #{'col1': [1, 2], 'col2': [3, 4]}
        #= pd.DataFrame(data=d)
        # df_interpolate = df_interpolate.append(, ignore_index=True)
        #ts.append(monthly_tsl)
        #new_ids.append(ids[i])


# Write merged DataFrames to csv files
print(df_interpolate)
df_interpolate.to_csv(data_path + 'TSL_interp-' + str(start_glacier) + '-' + str(end_glacier) + '.csv', sep=',')
