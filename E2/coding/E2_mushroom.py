# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:44:35 2019

@author: luni
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
import numpy as np
import pandas as pd
#import machine learning packages
from sklearn import linear_model, neighbors
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing, tree
from sklearn.compose import ColumnTransformer
#from sklearn.model_selection import train_test_split


#from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer,accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
#from sklearn.svm import LinearSVC
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle



######################################################################
#comments
#
#

######################################################################
#Input

filename = 'mushroom.csv'
methode = "forest"

cross_validation = 5

data_df = pd.read_csv("Data1/"+filename, sep=",",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)

#randomize data_df
#data_df = data_df.sample(frac=1)


data_df = shuffle(data_df)






######################################################################
#general preprocessing (not associated with any methode or any of the four preprocessing methods later on)

#veil-type ist bei jedem sample gleich, es gibt nur '1 Klasse', also kann dieses Attribut
#omitted werden

X = data_df.drop(columns = ['veil-type','poisonous or edible'])
y = data_df['poisonous or edible']


'''
y = y.replace(to_replace='p',value = 1)
y = y.replace(to_replace='e',value = 0)
'''


######################################################################
#preprocessing
#x1 ,x2 , x3 ,x4


#onehotencoded


X1 = X.copy()



enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(X1)
D1 = enc.transform(X1).toarray()

X1 = enc.transform(X1)


#mischung aus ordinal und hotencoding
X2 = X.copy()



X2['bruises'] = X2['bruises'].replace(to_replace='f',value=1)
X2['bruises'] = X2['bruises'].replace(to_replace='t',value=0)
X2['gill-attachment'] = X2['gill-attachment'].replace(to_replace='f',value=1)
X2['gill-attachment'] = X2['gill-attachment'].replace(to_replace='a',value=0)
X2['gill-spacing'] = X2['gill-spacing'].replace(to_replace='c',value=1)
X2['gill-spacing'] = X2['gill-spacing'].replace(to_replace='w',value=0)
X2['gill-size'] = X2['gill-size'].replace(to_replace='b',value=1)
X2['gill-size'] = X2['gill-size'].replace(to_replace='n',value=0)
X2['stalk-shape'] = X2['stalk-shape'].replace(to_replace='t',value=1)
X2['stalk-shape'] = X2['stalk-shape'].replace(to_replace='e',value=0)





enc = ColumnTransformer([("why", preprocessing.OneHotEncoder(handle_unknown='ignore'),\
                        ['cap-shape','cap-surface','cap-color','odor','gill-color','stalk-root',
                         'stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
                         'veil-color','ring-number','ring-type','spore-print-color','population',
                         'habitat\r'])],remainder='passthrough')
enc.fit(X2)
D2 = enc.transform(X2).toarray()

X2 = enc.transform(X2)










 






######################################################################
#applying methodes:

    #select methode
if type(methode) is str:
    if methode is "forest":
        #Tx, Ty = make_classification(n_samples=1000, n_features=4,
        #                             n_informative=2, n_redundant=0,
        #                             random_state=0, shuffle=False))
        
        
        methode = RandomForestClassifier(n_estimators = 100)
        #methode.fit(Tx,Ty)
    elif methode is "knn":
        k = 1
        weigh = "uniform"
        methode = neighbors.KNeighborsClassifier(k, weights=weigh)
    elif methode is"svm":
        methode = svm.SVC(gamma='scale')
    else:
        print("Error: Wrong methode chosen!")

######################################################################
#evaluation for all 5 measures (see pptx numeric_values: Slide 36):
    #fit X_train (moreless X1) with corresponding goal values y_train
    


score = {'accuracy': make_scorer(accuracy_score),
         'precision': make_scorer(precision_score, average='macro'),
         'recall': make_scorer(recall_score, average='macro')}
 






np.set_printoptions(precision = 3)

cv_results1 = cross_validate(methode, X1, y, scoring=score, cv=cross_validation)
for key1 in list(cv_results1.keys())[2:]:
    cv_results1[key1] = np.append(cv_results1[key1], cv_results1[key1].mean())
    print('{:35}'.format('evaluation for 1, ' + key1 + ": "), cv_results1[key1])
print("\n")
#Fit of 2
cv_results2 = cross_validate(methode, X2, y, scoring = score,  cv=cross_validation)
for key2 in list(cv_results2.keys())[2:]:
    cv_results2[key2] = np.append(cv_results2[key2], cv_results2[key2].mean())
    print('{:35}'.format('evaluation for 2, ' + key2 + ": "), cv_results2[key2])
print("\n")

#Fit of 3
'''
cv_results3 = cross_validate(methode, X3, y, scoring = score,  cv=cross_validation)
for key3 in list(cv_results3.keys())[2:]: 
    cv_results3[key3] = np.append(cv_results3[key3], cv_results3[key3].mean())
    print('{:35}'.format('evaluation for 3, ' + key3 + ": "), cv_results3[key3])
'''



'''
print("\n")
#Fit of 4 ..
cv_results4 = cross_validate(methode, X4, y, scoring=score,  cv=cross_validation)
for key4 in list(cv_results4.keys())[2:]:
    cv_results4[key4] = np.append(cv_results4[key4], cv_results4[key4].mean())
    print('{:35}'.format('evalution for 4, ' + key4 + ": "), cv_results4[key4])

'''


#Fit of 4 ..






