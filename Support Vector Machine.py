# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:50:25 2020

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
'SGDRegressor'

#Model fitting    
from sklearn.svm import LinearSVR

model = LinearSVR()

#Model parameters
param_grid = [
            {
            "epsilon":[0.5],
            "C":[0.5]# maximum allowed error
             }
            ]
"performing grid search and evaluation"
scores_df = gs(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = True)

print("--- %s seconds ---" % (datetime.now() - start_time))

'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

