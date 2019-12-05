
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
import numpy as np
from math import sqrt
import pandas as pd
#import machine learning packages
from sklearn import linear_model, neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing, tree
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


#from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
#from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
#from sklearn.svm import LinearSVC
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier



######################################################################
#comments
#
#

######################################################################
#Input

filename = 'congress_train.csv'
methode = "knn"

cross_validation = 5

data_df = pd.read_csv("E2/data/"+filename, sep=",",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)

######################################################################
#general preprocessing (not associated with any methode or any of the four preprocessing methods later on)

#ID dropped, since it contains no relevant information
data_df = data_df.drop(columns=['ID'])

#ordinal encoded
data_df=data_df.replace(to_replace='y',value=1)
data_df=data_df.replace(to_replace='n',value=0)
data_df=data_df.replace(to_replace='unknown',value=0.5)
data_df = data_df.replace(to_replace='democrat',value = 1)
data_df = data_df.replace(to_replace='republican',value = 0)




# get target value
y = data_df['class']
# drop unimportant columns
X = data_df.drop(columns=['class'])

######################################################################
#calculate Correlation
corr = data_df.corr()
corr_feature = corr['class'].sort_values(ascending=False)
#print(corr_feature)
######################################################################
#prediction calculation (20 times)

#counter which Preprocessing Type wins the most and which loses the most



######################################################################
#preprocessing
#x1  , x2 , x3 , x4 

X1 = data_df.copy()




X2 = data_df.copy()





#feature selection: die mit niedrigster Korrelation werden eliminiert: hier bei <6% sind das
#erstaunlicher Weise die Spalten immigration und water-project-sharing-cost
X3 = data_df.copy()
#X3 = X3[["adoption-of-the-budget-resolution",'physician-fee-freeze']]   #.drop(columns=['immigration','water-project-cost-sharing'])


#faszinierenderweise funktioniert der code, es werden spalten gelöscht und richtig
#berechnet, es kommt genau dasselbe raus wie wenn sie nicht gedroppt werden
X3 = X3.drop(columns=['immigration','water-project-cost-sharing'])




X4 = data_df.copy()


#Anmerkung: An Korrelationsmatrix sieht man, dass Klassen mit gewissen Attributen zu 90% korrellieren können
#Vermutung: Decision-Trees gut für so eine Aufgabe
#
C= data_df.corr()
C = C['class'].sort_values()














######################################################################
#applying methodes:

    #select methode
if type(methode) is str:
    if methode is "tree":
        #Tx, Ty = make_classification(n_samples=1000, n_features=4,
        #                             n_informative=2, n_redundant=0,
        #                             random_state=0, shuffle=False))
        
        
        methode = RandomForestClassifier(max_depth=2, random_state=0)
        #methode.fit(Tx,Ty)
    elif methode is "knn":
        k = 5
        weigh = "uniform"
        methode = neighbors.KNeighborsClassifier(k, weights=weigh)
    elif methode is"svm":
        methode = svm.SVC(gamma='scale')
    else:
        print("Error: Wrong methode chosen!")

######################################################################
#evaluation for all 5 measures (see pptx numeric_values: Slide 36):
    #fit X_train (moreless X1) with corresponding goal values y_train
    




score = ['accuracy', 'precision', 'recall']
np.set_printoptions(precision = 7)




#Fit of 1
cv_results = cross_validate(methode, X1, y, scoring = score,  cv=cross_validation)

print('test accuracy for 1: ', cv_results['test_accuracy'])  
print('test precision for 1: ', cv_results['test_precision']) 
print('test recall for 1: ', cv_results['test_recall'])



#Fit of 2
cv_results = cross_validate(methode, X2, y, scoring = score,  cv=cross_validation)

print('test accuracy for 2: ', cv_results['test_accuracy'])  
print('test precision for 2: ', cv_results['test_precision']) 
print('test recall for 2: ', cv_results['test_recall'])



#Fit of 3
cv_results3 = cross_validate(methode, X3, y, scoring = score,  cv=cross_validation)
                         
print('test accuracy for 3: ', cv_results3['test_accuracy'])  
print('test precision for 3: ', cv_results3['test_precision']) 
print('test recall for 3: ', cv_results3['test_recall'])


#Fit of 4 ..
















    
    
    
#%%
