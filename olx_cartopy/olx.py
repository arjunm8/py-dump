import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#cordinate reference system
import cartopy.crs as ccrs
import cartopy.feature as cfeature

user_data = pd.read_csv("user_data.csv")

#create projection of map(dimensions and distortion stuff)
proj = ccrs.Mercator()

plt.figure(figsize=(15,15))
#creating axis based on that projection
ax = plt.axes(projection=proj)
#sets the extents of the map to which data will be plotted
#ax.set_extent((-50.0,20.0,52.0,10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS,linestyle=':')
devices = user_data.channel.unique()
colors = ['r','g','b']
for i in range(len(devices)):
    ix = user_data.channel == devices[i]    
    x, y = user_data.user_long[ix],user_data.user_lat[ix]
#    print(devices[i]+" : "+str(len(x)))
#use plt.scatter(x,y,color,marker) for scatterplots, needs separate args for color and marker
ax.plot(x,y,colors[i]+'.',transform=ccrs.Geodetic(),label = devices[i])
plt.legend(loc="upper left")
