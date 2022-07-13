# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:24:05 2020

@author: 20044
"""

import numpy as np 
import pandas as pd
from nga_codes.general import *
from nga_codes.dataManagement import *
from nga_codes.evaluation import *
from datetime import datetime
from scipy.stats import describe
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
start_time = datetime.now()
# dataset preparation
'loading data'
df = load_data("immoData_v46.csv", downcasting = True)
'dataset preparation'
df = data_prep(df)
'data split and tranformation'
X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, X_train, X_train_trans, y_train = data_transformation(df)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'LINEAR REGRESSION'

#Model fitting    
from sklearn.linear_model import LinearRegression
model = LinearRegression()
    
#Model parameters
'setting parameter groups'
param_grid = [{'normalize': [False]}]

"performing grid search and evaluation"
scores_df = gs(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names)

print("--- %s seconds ---" % (datetime.now() - start_time))

'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


