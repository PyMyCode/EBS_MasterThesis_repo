# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 19:18:30 2020

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
X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, X_train, X_train_trans, y_train = data_transformation(df, feature_selection = False)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'SGDRegressor'

#Model fitting    
from sklearn.linear_model import SGDRegressor
model = SGDRegressor( loss="squared_loss" ,tol = 0.00001, max_iter = 10000, eta0 = 0.001, penalty = "elasticnet") # cost function is ordinary least square (MSE)

#Model parameters
param_grid =    [
            { #maximum epochs
             'alpha': [0.0001],
            "l1_ratio": [0.1]
           }
             ]
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
