# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:37:39 2019

@author: luni
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



#ACHTUNG!! erwartet eine csv datei, dessen erste Zeile aus den Attributnamen besteht, also 'poisonous or edible','cap-shape',....
#das ursrpüngliche Dataset hat das nicht, dann funktioniert der ganze code nicht!
data = pd.read_csv("mushroomh.csv")




k = 1


#ändere figsize um das bild kleiner oder größer zu machen
fig = plt.figure(figsize=(30,12))

#horizontaler, bzw vertikaler Abstand der plots
fig.subplots_adjust(hspace=0.4,wspace = 0.001)



for i in data:
    a1 = data[i]
    a2 = a1.value_counts()
    
    plot = fig.add_subplot(4, 6, k)
    
    #ändere Farbe für alle bars
    g1 = plt.bar(x = np.arange(len(a2.index)), height = a2.values, width=0.5, color = 'darkorange', alpha = 0.5)
    
    #zweite input dieser Funktion bestimmt die Labels der x-Achsen der einzelnen Plots
    plt.xticks(np.arange(len(a2.index)),a2.index)
    
    
    #Prozenttext des ersten Balkens
    b1 = int(float(a2.values[0])/8124*100)
    s1 = '{0}%'.format(b1)
    plot.text(0,a2.values[0]/2, s1 ,fontsize=12, ha='center')#, fontweight="bold")
    
    if len(a2.index)>=2:
        
        #Prozenttext des zweiten Balkens
        b2 = int(float(a2.values[1])/8124*100)
        s2 = '{0}%'.format(b2)
        plot.text(1,a2.values[1]/2, s2 ,fontsize=12, ha='center')#, fontweight="bold")
    
    if len(a2.index)>=3:
        
        #Prozenttext des dritten Balkens
        b3 = int((float(a2.values[2])/8124)*100)
        #will 'zahl%' als string bekommen, in python 3 geht das scheinbar nur noch so:
        s3 = '{0}%'.format(b3)
        plot.text(2,a2.values[2]/2, s3 ,fontsize=13, ha='center')
    
    
    
    #leere y-Achsenlabels, x-Achsen Titel und Mindestbreite der Diagramme
    plt.yticks([])
    plt.title(i)
    plt.xlim((-1, max(4,len(a2.index)+1)))
        
    k = k+1
    



plt.show

plt.savefig('shrooms2.png')













