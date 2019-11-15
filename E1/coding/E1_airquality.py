"""
.. module:: airquality.py
    :platform:  Windows
    :synopsis: preprocesses, and analyses AirQuality.csv, all 4 methodes (treeRegression, LinearRegression,             LassoRegression and K-nn) are performed at the end 5 measures for the prediction are being calculated.

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
from sklearn.impute import SimpleImputer

######################################################################

#comments
# "goal"-value is benzen value
# after check with correlation matrix, we decide to remove time and date

######################################################################
#Input

filename='AirQuality.csv'
methode = "tree"

air_df = pd.read_csv("E1/data/" + filename, sep =";", lineterminator="\n", encoding="utf-8",error_bad_lines=False)

######################################################################
#general preprocessing (not associated with any methode or any of the four preprocessing methods later on)
air_df = air_df[:9357]           
air_df=air_df.drop(columns=['NMHC(GT)','CO(GT)','NOx(GT)','NO2(GT)','Unnamed: 15','Unnamed: 16'])
hours=air_df['Time'].apply(lambda x: int(str(x).split('.')[0]))
air_df=air_df.drop(columns=['Time'])
air_df['Time']=hours
months=air_df['Date'].apply(lambda x: (str(x).split('/')[1]+'_'+str(x).split('/')[2]))
air_df=air_df.drop(columns=['Date'])
air_df['Date']=months
air_df=air_df.replace(['03_2004','04_2004','05_2004','06_2004','07_2004','08_2004','09_2004','10_2004','11_2004','12_2004','01_2005','02_2005','03_2005','04_2005'],range(14))

air_df['T']=pd.to_numeric(air_df['T'].str.replace(',', '.', regex=False))
air_df['RH']=pd.to_numeric(air_df['RH'].str.replace(',', '.', regex=False))
air_df['AH']=pd.to_numeric(air_df['AH'].str.replace(',', '.', regex=False))
air_df['C6H6(GT)']=pd.to_numeric(air_df['C6H6(GT)'].str.replace(',', '.', regex=False))

X=air_df.drop(columns=['C6H6(GT)'])
y = air_df['C6H6(GT)']

######################################################################
#calculate Correlation matrix
corr = air_df.corr()
corr_feature = corr['C6H6(GT)'].sort_values(ascending=False)
#print(corr_feature)

######################################################################
#prediction calculation (20 times)

#counter which Preprocessing Type wins the most and which loses the most
counter_win = [0,0,0,0]
counter_lose = [0,0,0,0]

for i in range(20): #mache 5 runs, jeweils unterschiedliche test und trainingsdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

######################################################################
#preprocessing
    #x1 = standard, x2 normalized, x3 minmax, x4 normalized -werte
    min_max= preprocessing.MinMaxScaler()
    minmax2 = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    scaler2 = preprocessing.StandardScaler()
    #input_nan = SimpleImputer(missing_values=-200, strategy='mean')
    
    X1 = X_train.copy()  #X1 minimum effort encoding
    
    scaler.fit(X1)
    X2 = scaler.transform(X1) #X2 zscore scaling
    
    min_max.fit(X1)
    #input_nan.fit(X1)
    #X3 = input_nan.transform(X1)
    X3 = min_max.transform(X1) #X3 minmax scaling
    
    X4=X_train.copy()
    X4=X4.drop(columns=['Time','Date'])
    scaler2.fit(X4)
    X4=scaler2.transform(X4)    
    
    X1_test=X_test.copy()  #test data zu X1
    
    X2_test=scaler.transform(X_test) #test data zu X2
    
    #X3_test = input_nan.transform(X1_test)
    X3_test=min_max.transform(X1_test) #test data zu X3

    X4_test=X1_test.copy()
    X4_test=X4_test.drop(columns=['Time','Date'])
    X4_test=scaler2.transform(X4_test)

######################################################################
#applying methodes:
    
    #select methode
    if type(methode) is str:
        if methode is "lasso":
            methode = linear_model.Lasso()    
        elif methode is "knn":
            k=10
            weigh ="uniform"
            methode=neighbors.KNeighborsRegressor(k, weights=weigh)
            y_test=np.array(y_test)  
        elif methode is "tree":
            methode = tree.DecisionTreeRegressor()   
        elif methode is"linear":
            methode = linear_model.LinearRegression()
        else:
            print("Error: Wrong methode chosen!")

######################################################################
#  evaluation for all 5 measures (see pptx numeric_values: Slide 36):
    #fit X_train (moreless X1) with corresponding goal values y_train
    methode.fit(X1,y_train)    
    y1 = np.array(methode.predict(X1_test))       
    #y1-y4, y_test
    # y1 is result of prediction y_test is "faked" goal value of prediction
    mittel = y_test.mean()
    mlist = [mittel for i in range(len(y_test))]
    rmse1 = sqrt(mean_squared_error(y_test,y1))
    mae1 = mean_absolute_error(y_test,y1)
    rrse1 = sqrt(sum(np.multiply(y1-y_test,y1-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae1 = sum(abs(y1-y_test))/sum(abs(y_test-mlist))
    pbar1 = [y1.mean() for i in range(len(y1))]
    spa1 = sum((y1-pbar1)*(y_test-mlist))/(len(y1)-1)
    sp1 = sum((y1-pbar1)*(y1-pbar1))/(len(y1)-1)
    sa = sum((y_test-mlist)*(y_test-mlist))/(len(y_test)-1)
    cor1 = spa1/sqrt(sp1*sa)

    methode.fit(X2,y_train)
    y2 = np.array(methode.predict(X2_test))  
    
    rmse2 = sqrt(mean_squared_error(y_test,y2))
    mae2 = mean_absolute_error(y_test,y2)
    rrse2 = sqrt(sum(np.multiply(y2-y_test,y2-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae2 = sum(abs(y2-y_test))/sum(abs(y_test-mlist))
    pbar2 = [y2.mean() for i in range(len(y2))]
    spa2 = sum((y2-pbar2)*(y_test-mlist))/(len(y2)-1)
    sp2 = sum((y2-pbar2)*(y2-pbar2))/(len(y2)-1)
    cor2 = spa2/sqrt(sp2*sa)

    methode.fit(X3,y_train)
    y3 = np.array(methode.predict(X3_test))
    
    rmse3=  sqrt(mean_squared_error(y_test,y3))
    mae3 = mean_absolute_error(y_test,y3)
    rrse3 = sqrt(sum(np.multiply(y3-y_test,y3-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae3 = sum(abs(y3-y_test))/sum(abs(y_test-mlist))
    pbar3 = [y3.mean() for i in range(len(y3))]
    spa3 = sum((y3-pbar3)*(y_test-mlist))/(len(y3)-1)
    sp3 = sum((y3-pbar3)*(y3-pbar3))/(len(y3)-1)
    cor3 = spa3/sqrt(sp3*sa)

    methode.fit(X4,y_train)
    y4 = np.array(methode.predict(X4_test))

    rmse4 = sqrt(mean_squared_error(y_test,y4))
    mae4 = mean_absolute_error(y_test,y4)
    rrse4 = sqrt(sum(np.multiply(y4-y_test,y4-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae4 = sum(abs(y4-y_test))/sum(abs(y_test-mlist))
    pbar4 = [y4.mean() for i in range(len(y4))]
    spa4 = sum((y4-pbar4)*(y_test-mlist))/(len(y4)-1)
    sp4 = sum((y4-pbar4)*(y4-pbar4))/(len(y4)-1)
    cor4 = spa4/sqrt(sp4*sa)

    entry=[rmse1, rmse2,  rmse3, rmse4 ]  
    #print(entry) 
    win=min(entry)
    lose=max(entry)
    print("rmse: "+"{:7.3f}".format(win), "   Preprocessing winner: ", entry.index(win)+1)       
    counter_win[entry.index(win)] +=1 
    counter_lose[entry.index(lose)] +=1 

    entry2=[rrse1, rrse2,  rrse3, rrse4]  
    #print(entry) 
    win2=min(entry2)
    print("rrse: "+"{:7.3f}".format(win2), "   Preprocessing winner: ", entry2.index(win2)+1)

    entry3=[mae1, mae2, mae3, mae4 ]  
    #print(entry) 
    win3=min(entry3)
    lose3=max(entry3)
    print("mae: "+"{:8.3f}".format(win3), "   Preprocessing winner: ", entry3.index(win3)+1)
    counter_win[entry3.index(win3)] +=1 
    counter_lose[entry3.index(lose3)] +=1 

    entry4=[rae1, rae2,  rae3, rae4 ]  
    #print(entry) 
    win4=min(entry4)
    print("rae: "+"{:8.3f}".format(win4), "   Preprocessing winner: ", entry4.index(win4)+1)


    entry5=[cor1, cor2,  cor3, cor4 ]  
    #print(entry) 
    win5=max(entry5)
    lose5=min(entry5)
    print("cor: "+"{:8.3f}".format(win5), "   Preprocessing winner: ", entry5.index(win5)+1)
    counter_win[entry5.index(win5)] +=1 
    counter_lose[entry5.index(lose5)] +=1 

    print('\n')

    #train_score=methode.score(X1,y_train) #TODO: not working: because: Line 77 / 63 vs 67/68
    #test_score=lin_reg.score(X_test,y_test)
    #print(train_score, test_score)
print("counter_win: \n",counter_win)
print("best Preproccesing: ", counter_win.index(max(counter_win))+1)
print('\n')
print("counter_lose: \n", counter_lose)
print("worst Preproccessing: ", counter_lose.index(max(counter_lose))+1)

#%%