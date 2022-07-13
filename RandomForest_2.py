# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:48:56 2020

@author: 20044
"""

import numpy as np 
import pandas as pd
from nga_codes.general import *
from nga_codes.dataManagement import *
from nga_codes.evaluation import *
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
# dataset preparation
'loading data'
df = load_data("immoData_v37.csv", downcasting = True)
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
'dataset preparation'

'deleting geo data'
geo_data = ['plz', 'geo_bln', 'geo_krs', 'regio3', 'houseNumber', 'streetPlain']
df = df.drop(geo_data, axis=1)
'deleting scoutID'
df = df.drop("scoutId", axis=1)
'deleting scoutID'
df = df.drop("kreisfrei", axis=1)  
'deleting scoutID'
df = df.drop("GEN_help", axis=1)  
df = df.drop("gen_and_bln", axis=1)
'deleting scoutID'
df = df.drop("count", axis=1)
'baseRent'
df = df.drop("baseRent", axis=1)

#Multicollinearity
"""
Removing variables with high Multicollinearity. 
Removing based on correlation matrix
Removing the feature based on feature selection in RF and 
other scores in the regressions.

"""
df = df.drop("livingSpace", axis=1)
df = df.drop("yearConstructedRange", axis=1)

'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
'data tranformation'

label_attributes = ["baseRentPerSqm"]
remaining_attributes = ['baseRentRange']    
cat_attributes = ['typeOfFlat', 'interiorQual', 'condition', 'petsAllowed', 
               "heatingType", "firingTypes", "energyEfficiencyClass", 
               "date"]
num_attributes = ['serviceCharge', "yearConstructed", "lastRefurbish",
                "livingSpaceRange", "noRooms", "noParkSpaces",
               "floor",  "numberOfFloors", "thermalChar", "heatingCosts",
               "electricityBasePrice", "electricityKwhPrice", 'pop',
               'pop_density' ]    

bool_attributes = ["newlyConst", "hasKitchen", "balcony", "cellar", "garden", "lift"]

"rearrange the columns"
df_help = df[remaining_attributes + label_attributes + num_attributes + bool_attributes + cat_attributes]

#categorical
'get dummy variables'
df = pd.get_dummies(df_help, columns = cat_attributes)

'getting all column names and order'
complete_col = list(df.columns)
l = len(remaining_attributes + label_attributes + num_attributes + bool_attributes)
cat_attributes= complete_col[l:]
feature_names = complete_col[2:]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df ,df["baseRentRange"]):
    X_train = df.iloc[train_index][feature_names]
    y_train = df.iloc[train_index][label_attributes]
    X_test = df.iloc[test_index][feature_names]
    y_test = df.iloc[test_index][label_attributes]

#pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median"))
      ])

#combining Transformation   
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("bool", "passthrough", bool_attributes),
    ("cat", "passthrough", cat_attributes)
    ])
 
'train data fit and trans'
X_train_trans = full_pipeline.fit_transform(X_train)
'test data trans'
X_test_trans = full_pipeline.transform(X_test)

'converting to sparse matrix'
X_train_spa = sparse.csr_matrix(X_train_trans)
X_test_spa = sparse.csr_matrix(X_test_trans)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'Random Forest Regressor'
#Model fitting
from  sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs = -1, 
                              oob_score = True
                              )

param_grid = [
            {
            "n_estimators":[100],
            "max_features":["sqrt"]
             }
            ]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import os
from datetime import datetime 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd


#Model Fitting
grid_search = GridSearchCV(model, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)

#Model Fitting
'fitting the model to the training set'
start_time = datetime.now()
grid_search.fit(X_train_spa, y_train)
runtime = datetime.now() - start_time
print(runtime)

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
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    'r**2 score'
    r_2_score = r2_score(y_test, y_pred)   
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
    
'''

NOTES:
link: https://www.youtube.com/watch?v=4EOCQJgqAOY

        1. max_features
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