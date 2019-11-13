
#%%
import numpy as np
from math import sqrt
import datetime as dt
import pandas as pd
#import plotting packages
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.plotting import figure, output_file, show
from bokeh.transform import jitter
#import machine learning packages
from sklearn import linear_model, neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#energy assumptions:
#   roof is negativ because volume is constant therefor big roof -> square -> better heating
#                                                       small roof -> long rectangle -> worse heating


filename='energy.csv'

energy_df = pd.read_csv("E1/data/" + filename, sep =";",                                                                          lineterminator="\n", encoding="utf-8",error_bad_lines=False)
corr=energy_df.corr()

corr["Y1"].sort_values(ascending=False)
#analyse of dataset-->irrel features, verteilung, art der daten, wie preprocessen?, anzahl attributes, anzahl samples,
# art der ergebnisse  

y = energy_df["Y1"]
X = energy_df.drop(columns=["Y2","Y1"])
          
counter_win = [0,0,0,0]
counter_lose = [0,0,0,0]


for i in range(20): #mache 5 runs, jeweils unterschiedliche test und trainingsdaten

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    #preprocessing
    #x1 = standard, x2 normalize, x3 minmax, x4 -werte
    minmax= preprocessing.MinMaxScaler()
    minmax2 = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    scaler2 = preprocessing.StandardScaler()


    X1=X_train.copy()

    scaler.fit(X1)
    X2= scaler.transform(X1)
     #X2 minmax scaling
    
    minmax.fit(X1)
    X3 = minmax.transform(X1) #X3 minmax scaling
    
    X4=X_train.copy()
    X4=X4.drop(columns=['X6'])
    scaler2.fit(X4)
    X4=scaler2.transform(X4)

    X1_test=X_test.copy()

    X2_test = scaler.transform(X1_test)
    
    X3_test = minmax.transform(X1_test) #X3 minmax scaling

    X4_test=X_test.copy()
    X4_test=X4_test.drop(columns=["X6"])
    X4_test = scaler2.transform(X4_test)

##########################################################

    #lin reg
    #lin_reg = linear_model.LinearRegression()
  ###########################
    #knn 
    k=10
    weigh ="uniform"
    knn=neighbors.KNeighborsRegressor(k, weights=weigh)
    y_test=np.array(y_test)  
 #####################
    #lasso
    
    
    #tree



    #select methode
    methode = knn # 
##########################################################

    #evaluation *5    
    methode.fit(X1,y_train)    
    y1 = np.array(methode.predict(X_test))       
    #y1-y4, y_test
    
    mittel = y_test.mean()
    mlist = [mittel for i in range(len(y_test))]
    rmse1 = sqrt(mean_squared_error(y1-y_test,y1-y_test))
    mae1 = mean_absolute_error(y1,y_test)
    rrse1 = sqrt(sum(np.multiply(y1-y_test,y1-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae1 = sum(abs(y1-y_test))/sum(abs(y_test-mlist))
    pbar1 = [y1.mean() for i in range(len(y1))]
    spa1 = sum((y1-pbar1)*(y_test-mlist))/(len(y1)-1)
    sp1 = sum((y1-pbar1)*(y1-pbar1))/(len(y1)-1)
    sa = sum((y_test-mlist)*(y_test-mlist))/(len(y_test)-1)
    cor1 = spa1/sqrt(sp1*sa)
    

    methode.fit(X2,y_train)
    y2 = np.array(methode.predict(X2_test))  
    
    rmse2 = sqrt(mean_squared_error(y2-y_test,y2-y_test))
    mae2 = mean_absolute_error(y2,y_test)
    rrse2 = sqrt(sum(np.multiply(y2-y_test,y2-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae2 = sum(abs(y2-y_test))/sum(abs(y_test-mlist))
    pbar2 = [y2.mean() for i in range(len(y2))]
    spa2 = sum((y2-pbar2)*(y_test-mlist))/(len(y2)-1)
    sp2 = sum((y2-pbar2)*(y2-pbar2))/(len(y2)-1)
    cor2 = spa2/sqrt(sp2*sa)
    
    methode.fit(X3,y_train)
    y3 = np.array(methode.predict(X3_test))
    
    rmse3=  sqrt(mean_squared_error(y3-y_test,y3-y_test))
    mae3 = mean_absolute_error(y3,y_test)
    rrse3 = sqrt(sum(np.multiply(y3-y_test,y3-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    rae3 = sum(abs(y3-y_test))/sum(abs(y_test-mlist))
    pbar3 = [y3.mean() for i in range(len(y3))]
    spa3 = sum((y3-pbar3)*(y_test-mlist))/(len(y3)-1)
    sp3 = sum((y3-pbar3)*(y3-pbar3))/(len(y3)-1)
    cor3 = spa3/sqrt(sp3*sa)
    
    methode.fit(X4,y_train)
    y4 = np.array(methode.predict(X4_test))

    rmse4 = sqrt(mean_squared_error(y4-y_test,y4-y_test))
    mae4 = mean_absolute_error(y4,y_test)
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
    print(win, entry.index(win)+1)       
    counter_win[entry.index(win)] +=1 
    counter_lose[entry.index(lose)] +=1 

    entry2=[rrse1, rrse2,  rrse3, rrse4]  
    #print(entry) 
    win2=min(entry2)
    print(win2, entry2.index(win2)+1)

    entry3=[mae1, mae2, mae3, mae4 ]  
    #print(entry) 
    win3=min(entry3)
    lose3=max(entry3)
    print(win3, entry3.index(win3)+1)
    counter_win[entry3.index(win3)] +=1 
    counter_lose[entry3.index(lose3)] +=1 

    entry4=[rae1, rae2,  rae3, rae4 ]  
    #print(entry) 
    win4=min(entry4)
    print(win4, entry4.index(win4)+1)


    entry5=[cor1, cor2,  cor3, cor4 ]  
    #print(entry) 
    win5=max(entry5)
    lose5=min(entry5)
    print(win5, entry5.index(win5)+1)
    counter_win[entry5.index(win5)] +=1 
    counter_lose[entry5.index(lose5)] +=1 

    print('\n')
            
print("counter_win: \n",counter_win)
print("best Preproccesing: ", counter_win.index(max(counter_win))+1)
print('\n')
print("counter_lose: \n", counter_lose)
print("worst Preproccessing: ", counter_lose.index(max(counter_lose))+1)