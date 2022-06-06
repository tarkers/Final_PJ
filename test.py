import pandas as pd 
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
# teset file feature 
'''
id,name,latitude,longitude,address,city,state,zip,country,url,phone,categories
全沒有MISS之資料: id,latitude,longitude
少部分沒Miss: name,country
'''
# extract feature
df=pd.read_csv('./train.csv')
latitude=df['latitude'].values
longitude=df['longitude'].values
plt.plot(longitude,latitude, 'ro', markersize=2)
plt.show()
# test =match_files.loc[match_files['match'] == True]
# test.to_csv("match.csv",index=False)
# print(test.head())