# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:03:05 2019

@author: luni
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


#benutze energy_training.csv und test hat alle änderungen, dass python damit arbeiten kann.
#normalisiert wurde jedoch noch nicht!! Decisiontrees verwenden keine Abstände
#deswegen ist ein normalisieren des Datensatzes nicht notwendig (für mich) 
trainingdata = pd.read_csv("energy_training.csv")
testdata = pd.read_csv("energy_test.csv")




#konvertiere dataframe zu numpy array

trainingdataA = trainingdata.to_numpy()
testdataA = testdata.to_numpy()
A = trainingdataA[:,0:8]
B = trainingdataA[:,8]

A1 = testdataA[:,0:8]
B1 = testdataA[:,8]


#alternativ: tree.DecisionTreeRegressor(max_depth=15), als Parameter übergeben. 
#Dies schreibt vor wieviele Level der Baum hat. 
#Für Regression werden sonst in unserem Fall 691 Endknoten erstellt, einer für jeden Datenpunkt
aTree = tree.DecisionTreeRegressor()   
aTree =aTree.fit(A, B)

#Predicton Vektor
P = aTree.predict(A1)

#Differenzvektor
D = np.subtract(P,B1)

plt.figure(figsize=(15,10))
#plot der testwerte und der prediction
plt.plot(np.arange(77),P,'yo')
plt.plot(np.arange(77),B1,'o')


#plot der Differenz der Datensätze
plt.figure(figsize=(10,7))
plt.plot(np.arange(77),D)




#tree.plot_tree(aTree.fit(A, B))




