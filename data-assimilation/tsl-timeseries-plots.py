import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

# TODO
# Merge csv files
# Join information from RGI


df = pd.read_csv('data/gee-results/TSL-results-103-104.csv', header=0)
df['LS_DATE'] = pd.to_datetime(df.LS_SCENE.str[12:20])

# print(df['LS_SCENE'])
# print(df['LS_DATE'])
print(df)
print(df.size)
# https://machinelearningmastery.com/time-series-data-visualization-with-python/

# plot.scatter(df['LS_DATE'], df['TSL_ELEV'])
# plot.show()

df_full.plot(x='LS_DATE', y='TSL_ELEV', style='k.', markersize=0.2, alpha=0.5)
pyplot.show()


# data = [go.Scatter(x=df.LS_DATE, y=df.TSL_ELEV)]
# py.iplot(data)
