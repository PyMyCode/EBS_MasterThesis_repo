# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:12:03 2020

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
from sklearn.linear_model import SGDRegressor
model = SGDRegressor( loss="squared_loss" ,tol = 0.00001, max_iter = 10000, penalty=None) # cost function is ordinary least square (MSE)

#Model parameters
param_grid = [
            { #maximum epochs
             'eta0': [0.001]}# starting learning rate (size of the step)
            ]
"performing grid search and evaluation"
scores_df = gs(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = True)

print("--- %s seconds ---" % (datetime.now() - start_time))

'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

'quick checks'
"before"
nga_histogram(X_train["serviceCharge"])
X_train["serviceCharge"].describe()

nga_histogram(y_train)
y_train.describe()

"after"
nga_histogram(y_train_trans)
describe(y_train_trans)

'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
'''
NOTES:
Stochastic Gradient Descent picks a random instance in the training set at every step
and computes the gradients based only on that single instance. Obviously,
working on a single instance at a time makes the algorithm much faster
because it has very little data to manipulate at every iteration.

On the other hand, due to its stochastic (i.e., random) nature, this algorithm
is much less regular than Batch Gradient Descent: instead of gently
decreasing until it reaches the minimum, the cost function will bounce up
and down, decreasing only on average.

'''
