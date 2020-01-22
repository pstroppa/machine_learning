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
from sklearn import preprocessing, tree
from math import sqrt
import pandas as pd
from sklearn.utils import shuffle
#import machine learning packages
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score
import warnings
#from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
#from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
#from sklearn.svm import LinearSVC
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

######################################################################
#comments
#
#

######################################################################
#Input

filename = 'amazon_train.csv'
tfidf= 'tfidf.csv'
methode = "forest"

cross_validation = 5

data_df = pd.read_csv(filename, sep=",",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)

tfidf_df = pd.read_csv(tfidf, sep=";",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)
######################################################################
#general preprocessing (not associated with any methode or any of the four preprocessing methods later on)

#ID dropped, since it contains no relevant information
data_df = data_df.drop(columns=['ID'])
X_tfidf = tfidf_df.copy()
X_tfidf=X_tfidf.fillna(0)


minmax= preprocessing.MinMaxScaler()


data_df = shuffle(data_df)

# get target value
y = data_df['Class']
# drop unimportant columns
X = data_df.drop(columns=['Class'])

#X[X!=0]=1
###############################################
#prepro
A=pd.DataFrame()

for word in X:
    A[word]=X[word].groupby(X[word]).count()

    A[word]=float(750-A[word][0])/750
    
A = A.loc[0]    
  
lis = list(A.index)

booli1 = list(A[lis]<0.65)
booli2 = list(A[lis]>0.01)
booli3 = list(A[lis]<0.40)
booli4 = list(A[lis]>0.02)

booliX1=booli1 and booli2
booliX2=booli1 and booli4
booliX3=booli3 and booli2
booliX4=booli3 and booli4

print('prepro done')

######################################################################
#preprocessing
#x1  , x2 , x3 , x4 


X1 = X.iloc[:,booliX1]

#minmax.fit(X1)

#X1=minmax.transform(X1)
X2 = X.iloc[:,booliX3]

X3 = X_tfidf.iloc[:,booliX1] 

X4 = X_tfidf.iloc[:,booliX3]  




######################################################################
#applying methodes:

    #select methode
if type(methode) is str:
    if methode is "forest":
        #Tx, Ty = make_classification(n_samples=1000, n_features=4,
        #                             n_informative=2, n_redundant=0,
        #                             random_state=0, shuffle=False))
        
        #gute werte 3,1000,100  prepro: 0.98, 0.01 -->0.611
        #                       prepro: 0.97, 0.01 -->0.619
        #                       prepro: 0.97, 0.013 -->0.597
        #                       prepro: 0.7 0.01 -->0.6
        #                       prepro: 0.65, 0.01 -->0.619
        #  3, 1000, 10 prepro: 0.65, 0.01 -->                  
        methode = RandomForestClassifier(min_samples_leaf=3, max_features=1000, n_estimators=700)
        #methode.fit(Tx,Ty)
    elif methode is "knn":
        k = 3
        weigh = "distance"
        methode = KNeighborsClassifier(k, weights=weigh)
    elif methode is"svm":
        methode = svm.SVC(gamma='auto', kernel='linear')
    else:
        print("Error: Wrong methode chosen!")

######################################################################
#evaluation accuracy, precision, recall

score = {'accuracy': make_scorer(accuracy_score),
         'precision': make_scorer(precision_score, average='macro'),
         'recall': make_scorer(recall_score, average='macro')}
 
np.set_printoptions(precision = 3)


cv_results1 = cross_validate(methode, X1, y, scoring=score, cv=cross_validation, n_jobs=-1)
for key1 in list(cv_results1.keys())[:]:
    cv_results1[key1] = np.append(cv_results1[key1], cv_results1[key1].mean())
    print('{:35}'.format('evalution for 1, ' + key1 + ": "), cv_results1[key1])
print("\n")

#Fit of 2
cv_results2 = cross_validate(methode, X2, y, scoring = score,  cv=cross_validation, n_jobs=-1)
for key2 in list(cv_results2.keys())[:]:
    cv_results2[key2] = np.append(cv_results2[key2], cv_results2[key2].mean())
    print('{:35}'.format('evalution for 2, ' + key2 + ": "), cv_results2[key2])
print("\n")

#Fit of 3
cv_results3 = cross_validate(methode, X3, y, scoring = score,  cv=cross_validation, n_jobs=-1)
for key3 in list(cv_results3.keys())[:]: 
    cv_results3[key3] = np.append(cv_results3[key3], cv_results3[key3].mean())
    print('{:35}'.format('evalution for 3, ' + key3 + ": "), cv_results3[key3])
print("\n")


cv_results4 = cross_validate(methode, X4, y, scoring=score, cv=cross_validation, n_jobs=-1)
for key4 in list(cv_results2.keys())[:]:
    cv_results4[key4] = np.append(cv_results4[key4], cv_results4[key4].mean())
    print('{:35}'.format('evalution for 4, ' + key4 + ": "), cv_results4[key4])
print("\n")