# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:59:27 2019

@author: luni
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("mushroom.csv")


#F = data[:][1] geht nicht, da spaltenindex die namen der zeilen als strings erwartet (warum auch immer)
#F = data[:]['cap-shape'] #funktioniert
#allerdings funktioniert iloc, ähnlich wie in matlab
#G = data.iloc[1:15,1]
#wichtig!!! 1:10 geht von 1,2,3,...,9 und inkludiert 10 nicht!
#außerdem fängt python von 0 an zu indizieren

nominals_count = np.zeros((12,23))

#print(nominals_count)


#berechne p und e Häufigkeit
#+ alle Häufigkeiten aller attribute
#schlecht geschrieben und sogar notation gewechselt, macht zum glück keine probleme
for i in data[:]['poisonous or edible']:
    if i == 'p':
        nominals_count[0,0] = nominals_count[0,0] + 1
    elif i == 'e':
        nominals_count[1,0] = nominals_count[1,0] + 1


for i in data[:]['cap-shape']:
    if i == 'b':
        nominals_count[0,1] = nominals_count[0,1] + 1
    elif i == 'c':
        nominals_count[1,1] = nominals_count[1,1] + 1
    elif i == 'x':
        nominals_count[2,1] = nominals_count[2,1] + 1
    elif i == 'f':
        nominals_count[3,1] = nominals_count[3,1] + 1
    elif  i == 'k':
        nominals_count[4,1] = nominals_count[4,1] + 1
    elif  i == 's':
        nominals_count[5,1] = nominals_count[5,1] + 1



for i in data[:]['cap-surface']:
    if i == 'f':
        nominals_count[0,2] += 1
    elif i == 'g':
        nominals_count[1,2] += 1
    elif i == 'y':
        nominals_count[2,2] += 1
    elif i == 's':
        nominals_count[3,2] += 1
    
    
    
    
    
for i in data[:]['cap-color']:
    if i == 'n':
        nominals_count[0,3] += 1
    elif i == 'b':
        nominals_count[1,3] += 1
    elif i == 'c':
        nominals_count[2,3] += 1
    elif i == 'g':
        nominals_count[3,3] += 1
    elif i == 'r':
        nominals_count[4,3] += 1
    elif i == 'p':
        nominals_count[5,3] += 1
    elif i == 'u':
        nominals_count[6,3] += 1   
    elif i == 'e':
        nominals_count[7,3] += 1
    elif i == 'w':
        nominals_count[8,3] += 1    
    elif i == 'y':
        nominals_count[9,3] += 1
    elif i == 't':
        nominals_count[10,3] += 1   
    elif i == 'f':
        nominals_count[11,3] += 1

        

for i in data[:]['bruises']:
    if i == 't':
        nominals_count[0,4] += 1
    elif i == 'f':
        nominals_count[1,4] += 1
    




for i in data[:]['odor']:
    if i == 'a':
        nominals_count[0,5] += 1
    elif i == 'l':
        nominals_count[1,5] += 1
    elif i == 'c':
        nominals_count[2,5] += 1
    elif i == 'y':
        nominals_count[3,5] += 1
    elif i == 'f':
        nominals_count[4,5] += 1
    elif i == 'm':
        nominals_count[5,5] += 1
    elif i == 'n':
        nominals_count[6,5] += 1   
    elif i == 'p':
        nominals_count[7,5] += 1
    elif i == 's':
        nominals_count[8,5] += 1    
    

        
        
 
       

for i in data[:]['gill-attachment']:
    if i == 'a':
        nominals_count[0][6] += 1
    elif i == 'd':
        nominals_count[1][6] += 1
    elif i == 'f':
        nominals_count[2][6] += 1
    elif i == 'n':
        nominals_count[3][6] += 1
    
        
        
        
        
for i in data[:]['gill-spacing']:
    if i == 'c':
        nominals_count[0][7] += 1
    elif i == 'w':
        nominals_count[1][7] += 1
    elif i == 'd':
        nominals_count[2][7] += 1
     





for i in data[:]['gill-size']:
    if i == 'b':
        nominals_count[0][8] += 1
    elif i == 'n':
        nominals_count[1][8] += 1






for i in data[:]['gill-color']:
    if i == 'k':
        nominals_count[0][9] += 1
    elif i == 'n':
        nominals_count[1][9] += 1
    elif i == 'b':
        nominals_count[2][9] += 1
    elif i == 'h':
        nominals_count[3][9] += 1
    elif i == 'g':
        nominals_count[4][9] += 1
    elif i == 'r':
        nominals_count[5][9] += 1
    elif i == 'o':
        nominals_count[6][9] += 1   
    elif i == 'p':
        nominals_count[7][9] += 1
    elif i == 'u':
        nominals_count[8][9] += 1    
    elif i == 'e':
        nominals_count[9][9] += 1
    elif i == 'w':
        nominals_count[10][9] += 1
    elif i == 'y':
        nominals_count[11][9] += 1
    






for i in data[:]['stalk-shape']:
    if i == 'e':
        nominals_count[0][10] += 1
    elif i == 't':
        nominals_count[1][10] += 1





for i in data[:]['stalk-root']:
    if i == 'b':
        nominals_count[0][11] += 1
    elif i == 'c':
        nominals_count[1][11] += 1
    elif i == 'u':
        nominals_count[2][11] += 1
    elif i == 'e':
        nominals_count[3][11] += 1
    elif i == 'z':
        nominals_count[4][11] += 1
    elif i == 'r':
        nominals_count[5][11] += 1
    elif i == '?':      #missing im txt, vll problem?
        nominals_count[6][11] += 1   
    


for i in data[:]['stalk-surface-above-ring']:
    if i == 'f':
        nominals_count[0][12] += 1
    elif i == 'y':
        nominals_count[1][12] += 1
    elif i == 'k':
        nominals_count[2][12] += 1
    elif i == 's':
        nominals_count[3][12] += 1



for i in data[:]['stalk-surface-below-ring']:
    if i == 'f':
        nominals_count[0][13] += 1
    elif i == 'y':
        nominals_count[1][13] += 1
    elif i == 'k':
        nominals_count[2][13] += 1
    elif i == 's':
        nominals_count[3][13] += 1
    




for i in data[:]['stalk-color-above-ring']:
    if i == 'n':
        nominals_count[0][14] += 1
    elif i == 'b':
        nominals_count[1][14] += 1
    elif i == 'c':
        nominals_count[2][14] += 1
    elif i == 'g':
        nominals_count[3][14] += 1
    elif i == 'o':
        nominals_count[4][14] += 1
    elif i == 'p':
        nominals_count[5][14] += 1
    elif i == 'e':
        nominals_count[6][14] += 1
    elif i == 'w':
        nominals_count[7][14] += 1
    elif i == 'y':
        nominals_count[8][14] += 1
    



for i in data[:]['stalk-color-below-ring']:
    if i == 'n':
        nominals_count[0][15] += 1
    elif i == 'b':
        nominals_count[1][15] += 1
    elif i == 'c':
        nominals_count[2][15] += 1
    elif i == 'g':
        nominals_count[3][15] += 1
    elif i == 'o':
        nominals_count[4][15] += 1
    elif i == 'p':
        nominals_count[5][15] += 1
    elif i == 'e':
        nominals_count[6][15] += 1
    elif i == 'w':
        nominals_count[7][15] += 1
    elif i == 'y':
        nominals_count[8][15] += 1





for i in data[:]['veil-type']:
    if i == 'p':
        nominals_count[0][16] = nominals_count[0][16] + 1
    elif i == 'u':
        nominals_count[1][16] = nominals_count[1][16] + 1






for i in data[:]['veil-color']:
    if i == 'n':
        nominals_count[0][17] += 1
    elif i == 'o':
        nominals_count[1][17] += 1
    elif i == 'w':
        nominals_count[2][17] += 1
    elif i == 'y':
        nominals_count[3][17] += 1
    




for i in data[:]['ring-number']:
    if i == 'n':
        nominals_count[0][18] += 1
    elif i == 'o':
        nominals_count[1][18] += 1
    elif i == 't':
        nominals_count[2][18] += 1
    





for i in data[:]['ring-type']:
    if i == 'c':
        nominals_count[0][19] += 1
    elif i == 'e':
        nominals_count[1][19] += 1
    elif i == 'f':
        nominals_count[2][19] += 1
    elif i == 'l':
        nominals_count[3][19] += 1
    elif i == 'n':
        nominals_count[4][19] += 1
    elif i == 'p':
        nominals_count[5][19] += 1
    elif i == 's':
        nominals_count[6][19] += 1
    elif i == 'z':
        nominals_count[7][19] += 1
    





for i in data[:]['spore-print-color']:
    if i == 'k':
        nominals_count[0][20] += 1
    elif i == 'n':
        nominals_count[1][20] += 1
    elif i == 'b':
        nominals_count[2][20] += 1
    elif i == 'h':
        nominals_count[3][20] += 1
    elif i == 'r':
        nominals_count[4][20] += 1
    elif i == 'o':
        nominals_count[5][20] += 1
    elif i == 'u':
        nominals_count[6][20] += 1
    elif i == 'w':
        nominals_count[7][20] += 1
    elif i == 'y':
        nominals_count[8][20] += 1






for i in data[:]['population']:
    if i == 'a':
        nominals_count[0][21] += 1
    elif i == 'c':
        nominals_count[1][21] += 1
    elif i == 'n':
        nominals_count[2][21] += 1
    elif i == 's':
        nominals_count[3][21] += 1
    elif i == 'v':
        nominals_count[4][21] += 1
    elif i == 'y':
        nominals_count[5][21] += 1
    



for i in data[:]['habitat']:
    if i == 'g':
        nominals_count[0][22] += 1
    elif i == 'l':
        nominals_count[1][22] += 1
    elif i == 'm':
        nominals_count[2][22] += 1
    elif i == 'p':
        nominals_count[3][22] += 1
    elif i == 'u':
        nominals_count[4][22] += 1
    elif i == 'w':
        nominals_count[5][22] += 1
    elif i == 'd':
        nominals_count[6][22] += 1
  
    



p = plt.figure(figsize=(20,7))


#for schleife die die nominals_count matrix durchläuft und alle (23 * bis zu 12) bar plots ausführt
#uU abfragen ob nominals_count[m,n] == 0, falls plots zu langsam ist
 

for i in range(23):
    for j in range(12):
        plt.bar(x = 0+ 1*i, height = nominals_count[j,i], width=0.5, bottom = np.sum(nominals_count[0:j,i]))



#labels des ersten attributs

plt.text( 0 , 0 + nominals_count[0:1,0] / 2. , 'p', ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
  
plt.text( 0 , np.sum(nominals_count[0:1,0]) + nominals_count[1,0]/ 2. , 'e' , ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")

#...

#Achsen-labels
plt.ylabel('Attribute-frequency')
plt.xlabel('Attributes')
plt.title('shrooms')
plt.xticks(np.arange(0,23))

plt.show

plt.savefig('shrooms1.png')
#print(nominals_count[0:12,16])
#print(np.sum(nominals_count[0:12,1]))



#
#
#ab hier ignorieren!!
#
#



