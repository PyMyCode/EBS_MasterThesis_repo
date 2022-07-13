# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:24:10 2020

@author: 20044
"""

import numpy as np 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
from nga_codes.general import *
from nga_codes.dataManagement import *

#Spatial dataset
dataFile = r"\VG250_Gemeindegrenzen.shp"
dataFile2 = r"\gadm36_DEU_1.shp"
dataFolder = r"C:\Users\20044\Desktop\Thesis\Pricing Model\Dataset\Spatial\VG250_Gemeindegrenzen"
mun_original = gpd.read_file(dataFolder + dataFile)
bl_original = gpd.read_file(dataFolder + dataFile2)
bl = bl_original.to_crs(epsg = 3857)
mun = mun_original.copy()
bl = bl_original.copy()
df = load_data("immoData_v32.csv", downcasting = True)
#--------------------------------------------------------------------------------

ax = gdf2.to_crs(epsg=5243).plot(figsize=(20,20), color ="#CCCCCC", edgecolor = "white")
ax.axis("off")


'getting the counts of the number of listing per PLZ'

df["count"] = 1

df = df.groupby(by="plz").sum()["count"].reset_index()

gdf.plz = gdf.plz.astype(int)

gdf = gdf.merge(df_plz, on='plz', how='left')

ax = gdf.to_crs(epsg=4326).plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='count', cmap='hot', legend=True)
ax.axis("off")

#--------------------------------------------------------------------------------
'creating a count column'
df["count"] = 1

'grouping by municipality'
df_mun = df.groupby(by="regio3").sum()["count"].reset_index()

'creating a union'
df_union = mun.merge(df_mun, how='outer', left_on=['GEN'], right_on=["regio3"])

'unique data'

df_uni=df_union.loc[df_union['GEN'].isna()==True]

gdf_uni=df_union.loc[df_union["count"].isna()==True]

s = df_original.loc[df_original["scoutId"] == immo_id]

df_uni["count"].sum()


'geo_data'
nga_find(gdf, "plz", 8525)
'immo_data'
x = nga_find(df_original, "geo_plz", 1057)
len(x)

for plz in plz_list:
    df_original = df_original.drop(df_original.loc[df_original["plz"] == plz].index, axis=0)

#--------------------------------------------------------------------------------

bl.head()
mun.head()

'simple plotting'
bl.plot()
mun.plot()

'accessing the polygones'
mun.geometry

'returns the polygones'
mun.geometry[0]

print(mun.geometry[0])

'return the list of geodata'
mun.geometry[0:100]

poly=Polygon([(1,1),
              (2,1),
              (2,2),
              (1,2)
              ])

'making your own polygone'
poly.plot()

'getting the area'
mun.geometry[0].area

'CRS'
mun.crs
#<Projected CRS: EPSG:3857>
#Name: WGS 84 / Pseudo-Mercator
#units in meters
bl.crs
#<Geographic 2D CRS: EPSG:4326>
bl = bl.to_crs(epsg = 3857)

# Make a multi-layered plot
ax = mun.plot(figsize(15,10), linewidth=0.1)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)

#understanding the spatial relations

'getting all mun poly of Schleswig-Holstein'
SH_mun = mun.loc[mun["SN_L"]=="01", "geometry"].squeeze()
'complete subset'
SH_mun_com = mun.loc[mun["SN_L"]=="01"].squeeze()

'plotting it with the germany boarder'
ax = SH_mun.plot(figsize(15,10), linewidth=0.1)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)

'plotting the cities in Schleswig-Holstein border'
ax = SH_mun_com.loc[SH_mun_com["BEZ"] == "Stadt"].plot(figsize(15,10), linewidth=0.1)
SH_mun_com.plot(edgecolor = "k", facecolor = "none" ,ax=ax)


'filering out'
bl.query('NAME_1=="Schleswig-Holstein"')


#touch relation
'IMPORTANT'
Neumünster = SH_mun_com.loc[SH_mun_com["GEN"]=="Neumünster", "geometry"].squeeze()
mun_touch = SH_mun_com[SH_mun_com.touches(Neumünster)]

ax = mun_touch.plot(figsize(15,10), linewidth=0.1)
SH_mun.plot(edgecolor = "k", facecolor = "none" ,ax=ax)


#non spatial df Joints
df_original = load_data("immoData_v32.csv", downcasting = True)
df = df_original[['regio3', 'geo_bln', 'baseRentPerSqm', "baseRentRange"]]

df['log_baseRentPerSqm'] = np.log(df.baseRentPerSqm)
df['quantile'] = pd.qcut(df['baseRentPerSqm'], 10, labels=False)

mun_help = mun[["GEN", "BEZ", "SN_L", "geometry"]]


'merging'
'Note: keep gepdataframe on the left'
df_merge = mun_help.merge(df, left_on='GEN', right_on='regio3')

'getting a spatial representation of the rent'
ax = df_merge.loc[df_merge["baseRentPerSqm"]>10].plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='baseRentPerSqm', cmap='hot', legend=True)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)
ax.axis("off")

ax = df_merge.plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='log_baseRentPerSqm', cmap='GnBu', legend=True)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)
ax.axis("off")

#############SPATIAL ANALYSIS################################################

'Choropleth Maps'
sns.distplot(target)

'10 quartile distrbutions'
ax = df_merge.plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='quantile', cmap='GnBu', legend=True)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)
ax.axis("off")

'10 quartile distrbutions'
ax = df_merge.plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='quantile', cmap='OrRd', legend=True)
bl.plot(edgecolor = "k", facecolor = "none" ,ax=ax)
ax.axis("off")

#Spatial Autocorrelation
import esda
import pysal
from libpysal import weights
from esda.moran import Moran, Moran_Local

'Spatial Lag'
#1. Joint counts
'making a hessen subset of the data'
hessen_incorrect = df_merge.loc[df_merge["geo_bln"]=="Hessen"]
hessen = hessen_incorrect.loc[hessen_incorrect["SN_L"]=="06"]

ax = hessen.plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='quantile', cmap='OrRd', legend=True)
ax.axis("off")


'''
The global spatial autocorrelation focuses on the overall trend in the dataset and tells us if the degree of clustering int eh dataset.
In contrast, The local spatial autocorrelation detects variability and divergence in the dataset, which helps us identify hot spots and cold spots in the data.

'''
#Spatial Weights
'''
Spatial weights are how we determine the area’s neighborhood.
Contiguity means that two spatial units share a common border of non-zero length. 
Operationally, we can further distinguish between a rook and a queen criterion of contiguity,
in analogy to the moves allowed for the such-named pieces on a chess board.



'''
'calculate Queen contiguity spatial weights'
wq = weights.Queen.from_dataframe(hessen)
wq.transform = "R"

'spatial lag'
hessen["w_baseRentPerSqm"] = weights.lag_spatial(wq, hessen["baseRentPerSqm"])

ax = hessen.plot(figsize=(20,20), edgecolor = "#CCCCCC", linewidth=0.1, column='w_baseRentPerSqm', cmap='OrRd', legend=True)
ax.axis("off")

'Global Spatial Autocorrelation'

'''
In our case, this number provides information that there is a positive spatial autocorrelation in this dataset
'''
y = hessen["baseRentPerSqm"].to_numpy()
moran = Moran(y, wq)
moran.I
#Out[162]: 0.27658830112898214
moran.p_sim
#Out[177]: 0.001
'for stats sign P value needs to be below 5%..this is 0.1%..hence significant?'



'Local Spatial Autocorrelation'
# calculate Moran Local 
m_local = Moran_Local(y, wq)

##########regression##############################################################################################

df_original = load_data("immoData_v32.csv", downcasting = True)

df_original_short = df_original.loc[df_original["geo_bln"] == "Hessen"]

df = df_original_short.copy()

num_attributes = ['serviceCharge', "yearConstructed", "lastRefurbish",
                    "livingSpaceRange", "noRooms", "noParkSpaces",
                   "floor", "thermalChar", "heatingCosts",'regio3',
                   'baseRentPerSqm'
                   ]

df = df_original_short[num_attributes]

mun_help = mun[["GEN","geometry"]]

df_merge = mun_help.merge(df, left_on='GEN', right_on='regio3')

drop_help = ["GEN",'regio3']

df_merge = df_merge.drop(drop_help, axis =1)

X,y = nga_adjust(df_merge, "baseRentPerSqm") 

'calculate Queen contiguity spatial weights'
wq = weights.Queen.from_dataframe(df_merge)
'regularizing the weights'
wq.transform = "R"

'calculation the spatial lag of the attributes'
XW = X.copy()

df_merge["baseRentPerSqm_w"] = weights.lag_spatial(wq, df_merge["baseRentPerSqm"])

df_merge = df_merge.drop(['geometry'], axis =1)

X,y = nga_adjust(df_merge, "baseRentPerSqm")

'linear regression without spatial lag'

from sklearn.linear_model import LinearRegression

'base Case'
le_base = LinearRegression()

X_base = X.drop(['baseRentPerSqm_w'], axis =1)

from sklearn.model_selection import cross_validate

cv_results = cross_validate(le_base, X_base, y, cv=5)


















