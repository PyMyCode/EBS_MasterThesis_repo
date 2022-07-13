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
from datetime import datetime
from scipy.stats import describe
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
# dataset preparation
'loading data'
df = load_data("immoData_v46.csv", downcasting = True)

df = data_prep(df)
'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
'dataset preparation'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'Random Forest Regressor'
#Model fitting
from  sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs = -1, 
                              oob_score = True
                              )

param_grid = [
            {
            "n_estimators":[10],
            "max_features":["sqrt"]
             }
            ]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
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