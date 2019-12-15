
#%%
"""
.. module:: file1.py
    :platform:  Windows
    :synopsis:   please discribe

.. moduleauthor: Peter Stroppa, Sophie Rain, Lucas Unterberger

.. Overview of the file:
    1) comments
    2) Input
    3) general preprocessing
    4) calculate Correlation
    5) calculate prediction
    6) preprocessing
    7) applying methods
    8) evaluation
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from math import sqrt
import pandas as pd
#import machine learning packages
from sklearn import linear_model, neighbors
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.utils import shuffle
#import E2.coding.find_distribution_short as fds
######################################################################
#comments
#
#

######################################################################
#Input

filename = "wine_red.csv"
methode = "forest"
cross_validation = 5        
data_df = pd.read_csv("E2/data/" + filename, sep=";",
                      lineterminator="\n", encoding="utf-8", error_bad_lines=False)
data_df = shuffle(data_df)
#data_df= data_df.sample(frac=1)
######################################################################

# find distribution that best suits distribution of attributes:
#for attribute in wine_df.columns.tolist():
#    best_name, best_distri = fds.find_best_distribution(wine_df[attribute], False)
#    print(attribute, ":", best_name)

######################################################################
#general preprocessing (not associated with any methode or any of the four
# preprocessing methods later on)

data_df.columns
# get target value
y = data_df["quality"]
# drop unimportant columns
X = data_df.drop(columns=["quality"])

######################################################################
#calculate Correlation
corr = data_df.corr()
corr_feature = corr["quality"].sort_values(ascending=False)
top5_corr = abs(data_df.corr()["quality"]).sort_values(ascending=False)[1:6]

######################################################################
#preprocessing
#x1  , x2 , x3 , x4 

minmax = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()

X1 = data_df.copy()

scaler.fit(X1)
X2 = scaler.transform(X1)  # X2 normalise scaling

minmax.fit(X1)
X3 = minmax.transform(X1)  # X3 minmax scaling

X4 = X1.copy()

for col in X4.columns:
    if corr_feature[col] < 0:
        X4[col] = (max(X4[col])-X4[col])/max(X4[col])*abs(corr_feature[col])
    else:
        X4[col] = X4[col] / max(X4[col])*abs(corr_feature[col])

#X4["alcohol"] = ((X4["alcohol"])/max((X4["alcohol"]))*10).copy()
#X4["volatile acidiy"] = ((X4["volatile acidity"])/max((X4["volatile acidity"]))*10).copy()
#X4["sulphates"] = ((X4["sulphates"])/max((X4["sulphates"]))*10).copy()
#X4["citric acidiy"] = ((X4["citric acidity"])/max((X4["citric acidity"]))*10).copy()
#X4["total sulfur dioxide"] = ((X4["fixed acidity"])/max((X4["fixed acidity"]))*10).copy()

#X4["free sulfur dioxide"] = ((X4["free sulfur dioxide"])/max((X4["free sulfur dioxide"]))*5).copy()
#X4["fixed acidiy"] = ((X4["fixed acidity"])/max((X4["fixed acidity"]))*10).copy()
#X4["fixed acidiy"] = ((X4["fixed acidity"])/max((X4["fixed acidity"]))*10).copy()

######################################################################
#applying methodes:

#select methode
if type(methode) is str:
    if methode is "forest":
        methode = RandomForestClassifier(n_estimators=100, max_features=6)
        # like this best! prep 3 / 4 all 1
        #max_depth=10, min_samples_split=2, min_samples_leaf=4 did not change anything
        # min_samples_leaf= 100 seems to be best.
    elif methode is "knn":
        k = 1
        weigh = "uniform"
        methode = neighbors.KNeighborsClassifier(k, weights=weigh)
        # 2 splits not full 1 best prep 4
    elif methode is"svm":
        methode = svm.SVC(gamma="scale", kernel="poly", degree=4)
        # poly best with prep 4 and degree = 4 all 1
        # best prep =4
    else:
        print("Error: Wrong methode chosen!")

######################################################################

score = {'accuracy': make_scorer(accuracy_score),
         'precision': make_scorer(precision_score, average='macro'),
         'recall': make_scorer(recall_score, average='macro')}
 
np.set_printoptions(precision = 3)

#Fit of 1
cv_results1 = cross_validate(methode, X1, y, scoring=score, cv=cross_validation)
for key1 in list(cv_results1.keys())[:]:
    cv_results1[key1] = np.append(cv_results1[key1], cv_results1[key1].mean())
    print('{:35}'.format('evaluation for 1, ' + key1 + ": "), cv_results1[key1])
print("\n")
#Fit of 2
cv_results2 = cross_validate(methode, X2, y, scoring = score,  cv=cross_validation)
for key2 in list(cv_results2.keys())[:]:
    cv_results2[key2] = np.append(cv_results2[key2], cv_results2[key2].mean())
    print('{:35}'.format('evaluation for 2, ' + key2 + ": "), cv_results2[key2])
print("\n")

#Fit of 3
cv_results3 = cross_validate(methode, X3, y, scoring = score,  cv=cross_validation)
for key3 in list(cv_results3.keys())[:]: 
    cv_results3[key3] = np.append(cv_results3[key3], cv_results3[key3].mean())
    print('{:35}'.format('evaluation for 3, ' + key3 + ": "), cv_results3[key3])

print("\n")
#Fit of 4 ..
cv_results4 = cross_validate(methode, X4, y, scoring=score,  cv=cross_validation)
for key4 in list(cv_results4.keys())[:]:
    cv_results4[key4] = np.append(cv_results4[key4], cv_results4[key4].mean())
    print('{:35}'.format('evaluation for 4, ' + key4 + ": "), cv_results4[key4])
print("\n")












    
    
    
#%%
