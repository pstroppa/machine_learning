
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


filename='bikeSharing_train.csv'

bike_df = pd.read_csv("E1/data/" + filename, sep =",",                                                                          lineterminator="\n", encoding="utf-8",error_bad_lines=False)

bike_df = bike_df.set_index("id", drop=True)

            
#analyse of dataset-->irrel features, verteilung, art der daten, wie preprocessen?, anzahl attributes, anzahl samples,
# art der ergebnisse  

y = bike_df["cnt"]
X = bike_df.drop(columns=["dteday","cnt"])
          

for i in range(20): #mache 5 runs, jeweils unterschiedliche test und trainingsdaten

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    #preprocessing
    #x1 = standard, x2 minmax-werte, x3 minmax, x4 -werte
    minmax= preprocessing.MinMaxScaler()
    minmax2 = preprocessing.MinMaxScaler()

    X1=X_train.copy()

    X2=X1.drop(columns=['weekday',"holiday"])
    minmax.fit(X2)
    X2 = pd.DataFrame(minmax.transform(X2)) #X2 minmax scaling
    
    minmax2.fit(X1)
    X3 = minmax2.transform(X1) #X3 minmax scaling
    
    X4=X_train.copy()
    X4=X4.drop(columns=['weekday',"holiday"])
    

    X1_test=X_test.copy()

    X2_test=X_test.drop(columns=['weekday',"holiday"])
    X2_test = pd.DataFrame(minmax.transform(X2_test)) #X2 minmax scaling
    
    
    
    X3_test = minmax2.transform(X1_test) #X3 minmax scaling
    X4_test=X_test.copy()
    X4_test=X4_test.drop(columns=['weekday',"holiday"])


    lin_reg = linear_model.LinearRegression()
    
    #k=10
    #weigh ="uniform"
    #knn=neighbors.KNeighborsRegressor(k, weights=weigh)
    #y_test=np.array(y_test)  

    mittel = y_test.mean()
    mlist = [mittel for i in range(len(y_test))]

    #select methode
    methode = lin_reg

    #evaluation *5    
    methode.fit(X1,y_train)
    y1 = np.array(methode.predict(X_test))        
    diff1=sqrt(sum(np.multiply(y1-y_test,y1-y_test)))/len(y_test)
    zsr1=sqrt(sum(np.multiply(y1-y_test,y1-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa1=sum(abs(y1-y_test))/len(y_test)
    zar1=sum(abs(y1-y_test))/sum(abs(y_test-mlist))
    pbar1=[y1.mean() for i in range(len(y1))]
    spa1=sum((y1-pbar1)*(y_test-mlist))/(len(y1)-1)
    sp1=sum((y1-pbar1)*(y1-pbar1))/(len(y1)-1)
    sa=sum((y_test-mlist)*(y_test-mlist))/(len(y_test)-1)
    cor1=spa1/sqrt(sp1*sa)
    

    methode.fit(X2,y_train)
    y2 = np.array(methode.predict(X2_test))  
    diff2=sqrt(sum(np.multiply(y2-y_test,y2-y_test)))/len(y_test)
    zsr2=sqrt(sum(np.multiply(y2-y_test,y2-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa2=sum(abs(y2-y_test))/len(y_test)
    zar2=sum(abs(y2-y_test))/sum(abs(y_test-mlist))
    pbar2=[y2.mean() for i in range(len(y2))]
    spa2=sum((y2-pbar2)*(y_test-mlist))/(len(y2)-1)
    sp2=sum((y2-pbar2)*(y2-pbar2))/(len(y2)-1)
    cor2=spa2/sqrt(sp2*sa)
    
    methode.fit(X3,y_train)
    y3 = np.array(methode.predict(X3_test))
    diff3=sqrt(sum(np.multiply(y3-y_test,y3-y_test)))/len(y_test)
    zsr3=sqrt(sum(np.multiply(y3-y_test,y3-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa3=sum(abs(y3-y_test))/len(y_test)
    zar3=sum(abs(y3-y_test))/sum(abs(y_test-mlist))
    pbar3=[y3.mean() for i in range(len(y3))]
    spa3=sum((y3-pbar3)*(y_test-mlist))/(len(y3)-1)
    sp3=sum((y3-pbar3)*(y3-pbar3))/(len(y3)-1)
    cor3=spa3/sqrt(sp3*sa)
    
    methode.fit(X4,y_train)
    y4 = np.array(methode.predict(X4_test))
    diff4=sqrt(sum(np.multiply(y4-y_test,y4-y_test)))/len(y_test) 
    zsr4=sqrt(sum(np.multiply(y4-y_test,y4-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
    zaa4=sum(abs(y4-y_test))/len(y_test)
    zar4=sum(abs(y4-y_test))/sum(abs(y_test-mlist))
    pbar4=[y4.mean() for i in range(len(y4))]
    spa4=sum((y4-pbar4)*(y_test-mlist))/(len(y4)-1)
    sp4=sum((y4-pbar4)*(y4-pbar4))/(len(y4)-1)
    cor4=spa4/sqrt(sp4*sa)

    entry=[diff1, diff2,  diff3, diff4 ]  
    #print(entry) 
    win=min(entry)
    print(win, entry.index(win)+1)
    #best.append(win)
            
    
    
    entry2=[zsr1, zsr2,  zsr3, zsr4]  
    #print(entry) 
    win2=min(entry2)
    print(win2, entry2.index(win2)+1)
    
    entry3=[zaa1, zaa2, zaa3, zaa4 ]  
    #print(entry) 
    win3=min(entry3)
    print(win3, entry3.index(win3)+1)
    
    entry4=[zar1, zar2,  zar3, zar4 ]  
    #print(entry) 
    win4=min(entry4)
    print(win4, entry4.index(win4)+1)
    
    entry5=[cor1, cor2,  cor3, cor4 ]  
    #print(entry) 
    win5=max(entry5)
    print(win5, entry5.index(win5)+1)
    
    
    print('\n')
            
       #print(min(best), best.index(min(best))+1, weigh, i) 
       
       
       
            #zsa=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test)))/len(y_test)
            #zsr=sqrt(sum(np.multiply(y_pred-y_test,y_pred-y_test))/sum(np.multiply(y_test-mlist,y_test-mlist)))
            #zaa=sum(abs(y_pred-y_test))/len(y_test)
            #zar=sum(abs(y_pred-y_test))/sum(abs(y_test-mlist))
