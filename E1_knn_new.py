# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:10:42 2019

@author: sophi
"""

#Erkenntnis: uniform besser als distance--sehr häufig
# das beste k schwankt sehr stark, liegt aber oft um die 30
#welches preprocessing am besten? fuer weights=uniform und k=30 kommt 4 sehr häufig vor,
# 1 und 2 gleichermaßen häufig und 3 sehr selten
#unterschiedliche meastures?

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

abssq=[]
relabs=[]
absabs=[]
relsq=[]

bestabssq=[]
bestrelsq=[]
bestabsabs=[]
bestrelabs=[]

#weight aus uniform und distance wählen
filename='Student_train.csv'
students = pd.read_csv(filename, sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
            
#analyse of dataset-->irrel features, verteilung, art der daten, wie preprocessen?, anzahl attributes, anzahl samples,
# art der ergebnisse  

          
X = students.copy()
del X['id']
del X['Grade']
y = list(students['Grade'])

#cat=[True,False,False,True,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
#cat=[0,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]

for i in range(5): #mache 5 runs, jeweils unterschiedliche test und trainingsdaten

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    #preprocessing
    enc = ColumnTransformer([("wtf", OneHotEncoder(handle_unknown='ignore'), [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22])])
    enc2 = ColumnTransformer([("wtf", OneHotEncoder(handle_unknown='ignore'), [0,1,3,4,7,8,11,12,13,14,15,16])])

    #enc = OneHotEncoder( categorical_features=cat)
    min_max= preprocessing.MinMaxScaler()
    minmax2 = preprocessing.MinMaxScaler()
    enc.fit(X_train)
    #enc2 = OneHotEncoder(categorical_features=cat)

    X1 = enc.transform(X_train)  #X1 minimum effort encoding
    X2 = preprocessing.scale(X1) #X2 zscore scaling
    min_max.fit(X1)
    X3 = min_max.transform(X1) #X3 minmax scaling
    X4 = X_train.copy() #delete irrel attributes + minmax scaling
    del X4['famsize']
    del X4['reason']
    del X4['traveltime']
    del X4['romantic']
    del X4['Walc']
    del X4['goout']
    del X4['activities']
    del X4['Mjob']
    enc2.fit(X4)
    X4=enc2.transform(X4)
    minmax2.fit(X4)
    X4 = minmax2.transform(X4)

    X4_test = X_test.copy()
    X_test=enc.transform(X_test)  #test data zu X1
    X_zscore=preprocessing.scale(X_test) #test data zu X2
    X_minmax=min_max.transform(X_test) #test data zu X3
    
 #test data zu X4
    del X4_test['famsize']
    del X4_test['reason']
    del X4_test['traveltime']
    del X4_test['romantic']
    del X4_test['Walc']
    del X4_test['goout']
    del X4_test['activities']
    del X4_test['Mjob']
    X4_test = enc2.transform(X4_test)
    X4_test = minmax2.transform(X4_test)

    #knn mit k in [1,3,5,10]
    #for weigh in ['uniform','distance']:
    #   best=[]
    #   for k in range(40):
    #        k=k+1
    k=30
    weigh='uniform'
            
    knn=neighbors.KNeighborsRegressor(k, weights=weigh)
    y_test=np.array(y_test)  
    mittel=y_test.mean()
    mlist=[mittel for i in range(len(y_test))]
    
    knn.fit(X1,y_train)
    y1 = np.array(knn.predict(X_test))        
    diff1=sqrt(sum(np.multiply(y1-y_test,y1-y_test)))/len(y_test)
    zsr1=sqrt(sum(np.multiply(y1-y_test,y1-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa1=sum(abs(y1-y_test))/len(y_test)
    zar1=sum(abs(y1-y_test))/sum(abs(y_test-mlist))
    pbar1=[y1.mean() for i in range(len(y1))]
    spa1=sum((y1-pbar1)*(y_test-mlist))/(len(y1)-1)
    sp1=sum((y1-pbar1)*(y1-pbar1))/(len(y1)-1)
    sa=sum((y_test-mlist)*(y_test-mlist))/(len(y_test)-1)
    cor1=spa1/sqrt(sp1*sa)
    
    #evaluation *5
    knn.fit(X2,y_train)
    y2 = np.array(knn.predict(X_zscore))  
    diff2=sqrt(sum(np.multiply(y2-y_test,y2-y_test)))/len(y_test)
    zsr2=sqrt(sum(np.multiply(y2-y_test,y2-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa2=sum(abs(y2-y_test))/len(y_test)
    zar2=sum(abs(y2-y_test))/sum(abs(y_test-mlist))
    pbar2=[y2.mean() for i in range(len(y2))]
    spa2=sum((y2-pbar2)*(y_test-mlist))/(len(y2)-1)
    sp2=sum((y2-pbar2)*(y2-pbar2))/(len(y2)-1)
    cor2=spa2/sqrt(sp2*sa)
    
    knn.fit(X3,y_train)
    y3 = np.array(knn.predict(X_minmax))
    diff3=sqrt(sum(np.multiply(y3-y_test,y3-y_test)))/len(y_test)
    zsr3=sqrt(sum(np.multiply(y3-y_test,y3-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa3=sum(abs(y3-y_test))/len(y_test)
    zar3=sum(abs(y3-y_test))/sum(abs(y_test-mlist))
    pbar3=[y3.mean() for i in range(len(y3))]
    spa3=sum((y3-pbar3)*(y_test-mlist))/(len(y3)-1)
    sp3=sum((y3-pbar3)*(y3-pbar3))/(len(y3)-1)
    cor3=spa3/sqrt(sp3*sa)
    
    knn.fit(X4,y_train)
    y4 = np.array(knn.predict(X4_test))
    diff4=sqrt(sum(np.multiply(y4-y_test,y4-y_test)))/len(y_test) 
    zsr4=sqrt(sum(np.multiply(y4-y_test,y4-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa4=sum(abs(y4-y_test))/len(y_test)
    zar4=sum(abs(y4-y_test))/sum(abs(y_test-mlist))
    pbar4=[y4.mean() for i in range(len(y4))]
    spa4=sum((y4-pbar4)*(y_test-mlist))/(len(y4)-1)
    sp4=sum((y4-pbar4)*(y4-pbar4))/(len(y4)-1)
    cor4=spa4/sqrt(sp4*sa)

    entry=[k, weigh, diff1, diff2,  diff3, diff4 ]  
    #print(entry) 
    win=min(entry[2:6])
    print(k,weigh, win, entry.index(win)-1)
    #best.append(win)
            
    
    
    entry2=[k, weigh, zsr1, zsr2,  zsr3, zsr4 ]  
    #print(entry) 
    win2=min(entry2[2:6])
    print(k,weigh, win2, entry2.index(win2)-1)
    
    entry3=[k, weigh, zaa1, zaa2,  zaa3, zaa4 ]  
    #print(entry) 
    win3=min(entry3[2:6])
    print(k,weigh, win3, entry3.index(win3)-1)
    
    entry4=[k, weigh, zar1, zar2,  zar3, zar4 ]  
    #print(entry) 
    win4=min(entry4[2:6])
    print(k,weigh, win4, entry4.index(win4)-1)
    
    entry5=[k, weigh, cor1, cor2,  cor3, cor4 ]  
    #print(entry) 
    win5=max(entry5[2:6])
    print(k,weigh, win5, entry5.index(win5)-1)
    
    
    print('\n')
            
       #print(min(best), best.index(min(best))+1, weigh, i) 
       
       
       
            #zsa=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test)))/len(y_test)
            #zsr=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
            #zaa=sum(abs(y_pred-y_test))/len(y_test)
            #zar=sum(abs(y_pred-y_test))/sum(abs(y_test-mlist))

