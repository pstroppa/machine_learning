# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:01:22 2019

@author: luni
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, tree
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


filename = 'mushroom.csv'
methode = "knn"

data_df = pd.read_csv("Data1/"+filename, sep=",",
                        lineterminator="\n", encoding="utf-8", error_bad_lines=False)


#data_df = shuffle(data_df)



X = data_df.drop(columns = ['veil-type','poisonous or edible'])
y = data_df['poisonous or edible']





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
X2 = enc.transform(X2).toarray()




X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.20)



methode = KNeighborsClassifier(n_neighbors =3)


methode.fit(X_train,y_train)

y_predict = methode.predict(X_test)










