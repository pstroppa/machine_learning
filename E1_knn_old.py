# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:35:44 2019

@author: sophi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn import preprocessing

abssq=[]
relabs=[]
absabs=[]
relsq=[]

bestabssq=[]
bestrelsq=[]
bestabsabs=[]
bestrelabs=[]

#weight aus uniform und distance wählen
filename='Student_premineff.csv'
students = pd.read_csv(filename, sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
            
            
X = students.copy()
del X['Grade']
del X['id']
y = list(students['Grade'])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

 
for weigh in ['uniform','distance']:
       for k in range(20):
            k=k+1
            knn=neighbors.KNeighborsRegressor(k, weights=weigh)
            
            #minmax
            
            #prerocessing necessary-->done in Students_preprocessing
            #learning, without any special parameters
            knn.fit(X_train,y_train)
            
            #predicting
            y_pred = np.array(knn.predict(X_test))
            y_test=np.array(y_test)
            mittel=y_test.mean()
            mlist=[mittel for i in range(len(y_test))]
            
            mmsa=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test)))/len(y_test)
            mmsr=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
            mmaa=sum(abs(y_pred-y_test))/len(y_test)
            mmar=sum(abs(y_pred-y_test))/sum(abs(y_test-mlist))
            
            
            
            
            #z-score
            #filename='Student_prezscore.csv'
            #students = pd.read_csv(filename, sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
            
            
            X = students.copy()
            del X['Grade']
            del X['id']
            y = list(students['Grade'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            
            #prerocessing necessary-->done in Students_preprocessing
            #learning, without any special parameters
            knn.fit(X_train,y_train)
            
            #predicting
            y_pred = np.array(knn.predict(X_test))
            y_test=np.array(y_test)
            
            zsa=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test)))/len(y_test)
            zsr=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
            zaa=sum(abs(y_pred-y_test))/len(y_test)
            zar=sum(abs(y_pred-y_test))/sum(abs(y_test-mlist))
            
   
            
            #mineff
            #filename='Student_premineff.csv'
            #students = pd.read_csv(filename, sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
            
            
            X = students.copy()
            del X['Grade']
            del X['id']
            y = list(students['Grade'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            
            #prerocessing necessary-->done in Students_preprocessing
            #learning, without any special parameters
            knn.fit(X_train,y_train)
            
            #predicting
            y_pred = np.array(knn.predict(X_test))
            y_test=np.array(y_test)
            
            mesa=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/len(y_test))
            mesr=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
            meaa=sum(abs(y_pred-y_test))/len(y_test)
            mear=sum(abs(y_pred-y_test))/sum(abs(y_test-mlist))
            
            
            abssq.append([k,mmsa, zsa, mesa])
            bestabssq.append( min(mmsa, zsa, mesa))
            
            relsq.append([k,mmsr, zsr, mesr])
            bestrelsq.append( min(mmsr, zsr ,mesr))
            
            absabs.append([k,mmaa, zaa, meaa])
            bestabsabs.append(min(mmaa, zaa, meaa))
            
            relabs.append([k,mmar, zar, mear])
            bestrelabs.append(min(mmar, zar, mear))
            
            #consider built-in scaling
            
            """
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            """
            
       #for k in range(20):
       #     print(bestabssq[k], bestrelsq[k], bestabsabs[k], bestrelabs[k])
            
            
       minabssq=min(bestabssq)
       minrelsq=min(bestrelsq)
       minabsabs=min(bestabsabs)
       minrelabs=min(bestrelabs)
       
       print(minabssq, bestabssq.index(minabssq))
       print( minrelsq,bestrelsq.index(minrelsq))
       print(minabsabs, bestabsabs.index(minabsabs))
       print(minrelabs, bestrelabs.index(minrelabs))