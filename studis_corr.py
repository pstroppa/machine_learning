# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:01:29 2019

@author: sophi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

filename='Student_train.csv'
students = pd.read_csv(filename, sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
students=students.drop(columns='id')


students=students.replace(to_replace='GP',value=1)
students=students.replace(to_replace='MS',value=0)

students=students.replace(to_replace='F',value=1)
students=students.replace(to_replace='M',value=0)

students=students.replace(to_replace='LE3',value=1)
students=students.replace(to_replace='GT3',value=0)

students=students.replace(to_replace='T',value=1)
students=students.replace(to_replace='A',value=0)

students=students.replace(to_replace='at_home',value=0)
students=students.replace(to_replace='services',value=1)
students=students.replace(to_replace='other',value=2)
students=students.replace(to_replace='teacher',value=3)
students=students.replace(to_replace='health',value=4)

students=students.replace(to_replace='reputation',value=1)
students=students.replace(to_replace='home',value=0)
students=students.replace(to_replace='course',value=2)

students=students.replace(to_replace='mother',value=1)
students=students.replace(to_replace='father',value=1)

students=students.replace(to_replace='yes',value=1)
students=students.replace(to_replace='no',value=0)

students=students.replace(to_replace='R',value=1)
students=students.replace(to_replace='U',value=0)
#enc = ColumnTransformer([("wtf", OneHotEncoder(handle_unknown='ignore'), [1,2,4,5,6,9,10,11,12,16,17,18,19,20,21,22,23])])

#enc.fit(X)
#X = enc.transform(X)

#X=pd.DataFrame(X)

print(students.corr()['Grade'].sort_values(ascending=False))
corr=students.corr()['Grade']
print(corr[abs(corr)<0.1])