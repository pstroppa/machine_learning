
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
import pandas as pd
from sklearn.metrics.scorer import make_scorer,accuracy_score,precision_score,recall_score
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

######################################################################
#comments
#
#

######################################################################
#Input

filename = 'congress_train.csv'
methode = "svm"

cross_validation = 5

data_df = pd.read_csv("Data1/"+filename, sep=",",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)




A = data_df.copy()

A=A.drop(columns= ['ID'])
A=A.replace(to_replace='democrat',value=1)
A=A.replace(to_replace='republican',value=0)


A=A.replace(to_replace='y',value=1)
A=A.replace(to_replace='n',value=0)
A=A.replace(to_replace='unknown',value=0.5)

#data_df=data_df.replace(to_replace='unknown',value=0.5)

corr = A.corr()
corr_feature = corr["class"].sort_values(ascending=False)
top5_corr = abs(A.corr()["class"]).sort_values(ascending=False)[1:6]


#data_df = shuffle(data_df)

######################################################################
#general preprocessing (not associated with any methode or any of the four preprocessing methods later on)

#ID dropped, since it contains no relevant information
data_df = data_df.drop(columns=['ID'])

X1 = data_df.copy()
X1 = X1.drop(columns=['class'])






#get target value
y = data_df['class']
#drop unimportant columns
X = data_df.drop(columns=['class'])



X=X.replace(to_replace='y',value=1)
X=X.replace(to_replace='n',value=0)
X=X.replace(to_replace='unknown',value=0.5)



######################################################################

#preprocessing
#x1  , x2 , x3 , x4 

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(X1)
#D3 = enc.transform(X3).toarray()

X1 = enc.transform(X1)


X2 = X.copy()


#feature selection: die mit niedrigster Korrelation werden eliminiert: hier bei <6% sind das
#erstaunlicher Weise die Spalten immigration und water-project-sharing-cost
X3 = X.copy()
#X3 = X3[["adoption-of-the-budget-resolution",'physician-fee-freeze']]   #.drop(columns=['immigration','water-project-cost-sharing'])


#faszinierenderweise funktioniert der code, es werden spalten gelöscht und richtig
#berechnet, es kommt genau dasselbe raus wie wenn sie nicht gedroppt werden
X3 = X3.drop(columns=['immigration','water-project-cost-sharing'])


X4 = data_df.copy()

#ordinal encoded
X4=X4.replace(to_replace='y',value=1)
X4=X4.replace(to_replace='n',value=0)
X4=X4.replace(to_replace='unknown',value=0.5)

#X4 = X4.replace(to_replace='democrat',value = 1)
#X4 = X4.replace(to_replace='republican',value = 0)

X4 = X4.drop(columns=['class'])



for col in X4.columns:
    if corr_feature[col] < 0:
        X4[col] = (max(X4[col])-X4[col])/max(X4[col])*abs(corr_feature[col])
    else:
        X4[col] = X4[col] / max(X4[col])*abs(corr_feature[col])



#Anmerkung: An Korrelationsmatrix sieht man, dass Klassen mit gewissen Attributen zu 90% korrellieren können
#Vermutung: Decision-Trees gut für so eine Aufgabe
#
#C= data_df.corr()
#C = C['class'].sort_values()






######################################################################
#applying methodes:

    #select methode
if type(methode) is str:
    if methode is "forest":
        #Tx, Ty = make_classification(n_samples=1000, n_features=4,
        #                             n_informative=2, n_redundant=0,
        #                             random_state=0, shuffle=False))
        
        
        methode = RandomForestClassifier(n_estimators=100)
        #methode.fit(Tx,Ty)
    elif methode is "knn":
        k = 5
        weigh = "distance"
        methode = KNeighborsClassifier(k, weights=weigh)
    elif methode is"svm":
        methode = svm.SVC(kernel='poly',coef0 = 10,gamma='auto')
    else:
        print("Error: Wrong methode chosen!")

######################################################################
#evaluation for all 5 measures (see pptx numeric_values: Slide 36):
    #fit X_train (moreless X1) with corresponding goal values y_train
    




#score = ['accuracy', 'precision_weighted', 'recall_weighted']

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
cv_results3 = cross_validate(methode, X3, y, scoring = score,  cv=cross_validation)
for key3 in list(cv_results3.keys())[2:]: 
    cv_results3[key3] = np.append(cv_results3[key3], cv_results3[key3].mean())
    print('{:35}'.format('evaluation for 3, ' + key3 + ": "), cv_results3[key3])

print("\n")

cv_results4 = cross_validate(methode, X4, y, scoring = score,  cv=cross_validation)
for key4 in list(cv_results4.keys())[2:]: 
    cv_results4[key4] = np.append(cv_results4[key4], cv_results4[key4].mean())
    print('{:35}'.format('evaluation for 4, ' + key4 + ": "), cv_results4[key4])















    
    
    
#%%
