# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:17:57 2020

@author: 20044
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

filepath = "C:\\Users\\20044\\.spyder-py3\\Practice\\Webscraping\\Kaggle Data\\"

edapath = "C:\\Users\\20044\\.spyder-py3\\Practice\\Webscraping\\EDA\\"

###########################################
var = "baseRent"
###########################################

#loading the data
df = pd.read_csv(filepath + 'immo_data.csv')

df_original = df

#variable list
var_list = df.columns.values.tolist()

#creating a variable discription overview
df_desc = df.describe(include='all').transpose()
df_desc["Type"] = df.dtypes

#Save
df_desc.to_excel(edapath + "var_list.xlsx", index = True)


#Data counting

subset_df = df[df[var] == 0 ]
var_count = subset_df["scoutId"].count()

#conditional removal of data
df = df.query("baseRent > 0")
df = df.reset_index(drop=True)

#removing Super outliers
q_low = df[var].quantile(0.01)
q_hi  = df[var].quantile(0.99)

df = df[(df[var] < q_hi) & (df[var] > q_low)]

#histogram
sns.distplot(df[var]);

#skewness and kurtosis
print("Skewness: %f" % df[var].skew())
print("Kurtosis: %f" % df[var].kurt())

##NOMINAL Variables
#scatter plot 
var2 = 'baseRent'
data = pd.concat([df[var], df[var2]], axis=1)
data.plot.scatter(x=var2, y=var);

## Categorical variables
#box plot overallqual/saleprice
var2 = 'regio1'
data = pd.concat([df[var], df[var2]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var2, y=var, data=data)
fig.axis(ymin=0, ymax=3000);
plt.xticks(rotation=90);

