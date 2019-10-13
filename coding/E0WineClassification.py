from __future__ import unicode_literals
from future_builtins import map  
from collections import Counter
from itertools import chain
import csv
from math import log
import numpy 
from math import fsum
import matplotlib.pyplot as plt

"""Wine Quality"""

filename='wine_red.csv'

reader=csv.reader(open(filename))
count=-1                            
samples=[]                     #empty lsit


for row in reader:
    if count==-1: #avoid first line
        count=1
    
    else:
        instance=row[0]  #string containing one sample
        attr=instance.split(";")
        attr=[float(elem) for elem in attr] #type cast into floats
        samples.append(attr)
        

values=[0,0,0,0,0,0,0,0,0,0,0]

for elem in samples:
        values[int(elem[11])]=values[int(elem[11])]+1

for i in range(11):
    if values[i]<10:
        plt.annotate(str(values[i]), (i-0.12,int(values[i])+8))
    elif values[i]<100:
        plt.annotate(str(values[i]), (i-0.22,int(values[i])+8))
    else:
        plt.annotate(str(values[i]), (i-0.36,int(values[i])+8))       
plt.bar(range(11), values)
plt.axis([-0.5,10.5,0,750])
plt.title('Distribution Quality')
plt.ylabel('Number of samples')
plt.xlabel('Quality value')
plt.savefig('distribution_winequality.png')
plt.close

"""Mushrooms"""

filename='mushrooms.csv'

reader=csv.reader(open(filename))                           
samples=[]                     #empty lsit


for row in reader:
        instance=row[0]  #string containing one sample
        attr=instance.split(";")
        samples.append(attr)
        

values=[0,0]

for elem in samples:
    if elem[0]=='e':
        values[0]=values[0]+1
    else:
        values[1]=values[1]+1

"""for i in range(11):
    if values[i]<10:
        plt.annotate(str(values[i]), (i-0.12,int(values[i])+8))
    elif values[i]<100:
        plt.annotate(str(values[i]), (i-0.22,int(values[i])+8))
    else:
        plt.annotate(str(values[i]), (i-0.36,int(values[i])+8))"""       
plt.bar(['edible','poisonous'], values)
#plt.axis([-0.5,10.5,0,750])
plt.title('Distribution Mushrooms')
plt.ylabel('Number of samples')
plt.xlabel('Edibility')
plt.savefig('distribution_mushrooms.png')