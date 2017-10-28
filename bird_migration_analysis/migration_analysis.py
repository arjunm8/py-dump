# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:07:19 2017

@author: Arjun
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
#cordinate reference system
import cartopy.crs as ccrs
import cartopy.feature as cfeature

birddata = pd.read_csv("bird_tracking.csv")
#fetch unique entries
bird_names = pd.unique(birddata.bird_name)

'''
#flight plot

plt.figure(figsize=(5,5))
for bird_name in bird_names:
    #save bool array of birddata entries with birdname==eric
    ix = birddata.bird_name == bird_name
    x, y = birddata.longitude[ix],birddata.latitude[ix]
    plt.plot(x,y,"-",label=bird_name)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.legend(loc="lower right")
'''

'''
#speed hist with manual nan tackling

ix = birddata.bird_name == "Eric"
speed = birddata.speed_2d[ix]
#nan=nonnumeric, return bool array and complement to get bool array of numerics
ind = ~(np.isnan(speed))
plt.hist(speed[ind], bins = np.linspace(0,30,20),normed=True)
plt.xlabel("2D speed m/s")
plt.ylabel("Frequency")
'''

'''
#pandas inbuilt plotting autodeals with nans

birddata.speed_2d.plot(kind="hist",range=[0,30])
plt.xlabel("2d speed")
'''


'''
#deciphering timestamps
#counting days and stuff
timestamps = []
for k in range(len(birddata)):
    #strptime will format the string into datetime object
    timestamps.append(datetime.datetime.strptime\
                      (birddata.date_time.iloc[k][:-3],"%Y-%m-%d %H:%M:%S"))
birddata["timestamp"] = pd.Series(timestamps,index = birddata.index)
times = birddata.timestamp[birddata.bird_name=="Eric"]
#returns timedelta array(timedelta=timeelapsed/difference)
elapsed_time  = [time - times[0] for time in times]
#plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))
#plt.xlabel("observations")
#plt.ylabel("time elapsed in days")
elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)


#calculation of mean speed for each day
next_day = 1
inds = []
daily_mean_speeds = []
for (i,t) in enumerate(elapsed_days):
    if t<next_day:
        inds.append(i)
    else:
        #compute mean speed
        daily_mean_speeds.append(np.mean(birddata.speed_2d[inds]))
        next_day+=1
        inds.clear()
plt.figure(figsize=(8,6))
plt.plot(daily_mean_speeds)
plt.xlabel("days")
plt.ylabel("mean speed m/s")
'''

#create projection of map(dimensions and distortion stuff)
proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
#creating axis based on that projection
ax = plt.axes(projection=proj)
#has to be found through trial error dafuq
ax.set_extent((-25.0,20.0,52.0,10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS,linestyle=':')

colors = ["r","b","g"]

for iname in range(len(bird_names)):
    ix = birddata.bird_name == bird_names[iname]
    x, y = birddata.longitude[ix],birddata.latitude[ix]
    ax.plot(x,y,colors[iname]+'-',transform=ccrs.Geodetic(),label=bird_names[iname])

plt.legend(loc="upper left")


# First, use `groupby` to group up the data.
grouped_birds = birddata.groupby("bird_name")

# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()

# The `head` method prints the first 5 lines of each bird.
grouped_birds.head()

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = grouped_birds.altitude.mean()
