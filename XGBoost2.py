# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:02:19 2020

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
# dataset preparation
'loading data'
df = load_data("immoData_v46.csv", downcasting = True)
'dataset preparation'
df = data_prep(df)
'data split and tranformation'
X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, X_train, X_train_trans, y_train = data_transformation(df, feature_select = True)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'Random Forest Regressor'
#Model fitting
import xgboost as xgb

model = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.0468,
                             reg_alpha=0.5, reg_lambda=0.5,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# Train the model on training data

#Model parameters
param_grid = [
            {
            "max_depth":[20],
            "n_estimators":[100],
            "colsample_bytree":[0.5],
            "learning_rate":[0.05],
            "subsample":[0.8],
            "min_child_weight":[1]           
             }
            ]
start_time = datetime.now()
"performing grid search and evaluation"
#scores_df = gs_rf(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = True, parametric= False)
scores_df = gs(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = True, parametric= False)
runtime = datetime.now() - start_time
print(runtime)


def gs_rf(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = False):
    from datetime import datetime
    
    
    grid_search = GridSearchCV(model, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True)
    
    #Model Fitting
    'fitting the model to the training set'
    grid_search.fit(X_train_spa, y_train_trans)
    
    
    
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    #best_param = grid_search.best_params_ #the parameters of that gave the best estimations 
    cvres = grid_search.cv_results_   #complete cross validation results
    scores_df = pd.DataFrame(cvres)
    
    'feature importance'
    feature_importance = best_model.feature_importances_.tolist()
    result = list(sorted(zip(feature_importance, feature_names), reverse = True)) #printing the coeff of all features in the best param model
    
    
    'result output report'
    FORMAT = '%Y-%m-%d-%H-%M-%S'
    ts = str(datetime.now().strftime(FORMAT))
    dataFile = ts + ".txt"
    dataFolder = r"C:\Users\20044\Desktop\Thesis\Pricing Model\Code\Regression_Output"
    csv_path = os.path.join(dataFolder, dataFile)
        
    file_txt = open(csv_path,"w")
    
    "REPORT"
    
    
    file_txt.write("GRID SEARCH REPORT\n\n")
    
    'test scores'
    if test_scores == True:
            #Test evaluation
        'rmse'
        y_pred = best_model.predict(X_test_spa)
        mse = mean_squared_error(y_test_trans, y_pred)
        rmse = np.sqrt(mse)
        'r**2 score'
        r_2_score = r2_score(y_test_trans, y_pred)
        file_txt.write("Testing Scores:\n")
        file_txt.write('MSE:\t{}\n'.format(mse))
        file_txt.write('RMSE:\t{}\n'.format(rmse))
        file_txt.write('R^2:\t{:.15%}\n\n'.format(r_2_score))
    
    'training scores'
    file_txt.write("Training Scores:\n\n")
    file_txt.write('Base model:\n{}\n'.format(best_model))
    file_txt.write('Best params:\n{}\n'.format(best_params))
    file_txt.write('\nBest model score:\t{}\n\n'.format(np.sqrt(-best_score)))
    
    
    file_txt.write("Following are the model scores:\n")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file_txt.write('{}\t{}\n'.format(np.sqrt(-mean_score), params))

    "feature importance"
    file_txt.write("\nFeature Importance:\n")
    for mean_score, feat in result:
        file_txt.write('{}\t{}\n'.format(mean_score, feat))
    
    file_txt.close()
    
    return scores_df