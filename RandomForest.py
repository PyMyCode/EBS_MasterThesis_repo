# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:37:07 2020

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
df = load_data("immoData_v32.csv", downcasting = True)
'dataset preparation'
df = data_prep(df)
'data split and tranformation'
X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, X_train, X_train_trans, y_train = data_transformation(df)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'Random Forest Regressor'
#Model fitting    
from  sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs = -1)

# Train the model on training data

#Model parameters
param_grid = [
            {
            "max_depth":[20],
            "n_estimators":[10],
            "max_features":["auto"]
             }
            # {
            # "max_depth":[5],
            # "n_estimators":[10, 20, 30],
            # "max_features":["auto"]
            #  }
            ]
start_time = datetime.now()
"performing grid search and evaluation"
scores_df = gs_rf(model, param_grid, X_train_spa, y_train_trans, X_test_spa, y_test_trans, feature_names, test_scores = True)
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
    
    'Grid Search Scores'
    file_txt.write("GRID SEARCH REPORT\n")
    file_txt.write("Runtime:\t{}\n\n".format(runtime))
    
    'test scores'
    if test_scores == True:
            #Test evaluation
        'rmse'
        y_pred = best_model.predict(X_test_spa)
        mse = mean_squared_error(y_test_trans, y_pred)
        rmse = np.sqrt(mse)
        'r**2 score'
        r_2_score = r2_score(y_test_trans, y_pred)
        
        file_txt.write('RMSE:\t{}\n'.format(rmse))
        file_txt.write('R^2:\t{:.15%}\n\n'.format(r_2_score))
    

    file_txt.write('\nBest parameters score:\t{}\n\n'.format(np.sqrt(-best_score)))
    
    file_txt.write("Following are the model scores:\n")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file_txt.write('{}\t{}\n'.format(np.sqrt(-mean_score), params))

    "feature importance"
    file_txt.write("\nFeature Importance:\n")
    for mean_score, feat in result:
        file_txt.write('{}\t{}\n'.format(mean_score, feat))
    
    file_txt.close()
    
    return scores_df
'''
NOTES:
link: https://www.youtube.com/watch?v=4EOCQJgqAOY

        1. min_samples_split
when ever you are making a split(at every node of the decision tree). 
You take a subset of the features.
randomly sub sample k dimensions from the total d dimensions
only split on k<<d features 

take:
k = sqr(d)


        2. max_depth
Full throttle it down to the maximum to take in the high variance
        
        3. Feature engineering
No need of feature scaling, since it is just splitting. Scale of feature does not matter.

        4. n_estimators
Keep n AS LARGE AS POSSIBLE (computational limitations ofc!!)
Higher n.. higher "wisdom of the crowd effect"
Note: Size of the sampb,le dataset is same as the original dataset with replacement
D = [a,b,c]
d1 = [a,a,b]
d2 = [b,b,c]
sometimes because of repetition, some datapoints get more emphasis, some get lesser emphasis. etc.

        5. Feature selection
One of the best algorithms for feature selection
When ever you split of a certain feature, how much does the impurity go down? 
Then you rank the feature accordingly.

        6. Training Testing Data
                Out of Bag
You do not have to make training, testing split
you can directly estimate on the training dataset

the data that was not a part of the training dataset for each particular 
predictor becomes the oob for that dataset that can be used as a test score.

Hence it does not make sense to do a cross validation of the set

This is super powerful since you can train the algorithm on the entire dataset
and you are getting an truely unbiased testing mechanism.

###################################
Boosting





'''







