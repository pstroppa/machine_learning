"""Mushrooms"""

from __future__ import unicode_literals
from future.builtins import map  
from collections import Counter
from itertools import chain
import csv
from math import log
import numpy as np
from math import fsum
import matplotlib.pyplot as plt

filename='data/mushrooms.csv'

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

plt.annotate(str(values[0]), (-0.06,3900))    
plt.annotate(str(values[1]), (0.94,3600))
plt.bar(['edible','poisonous'], values)
#plt.axis([-0.5,10.5,0,750])
plt.title('Distribution Mushrooms')
plt.ylabel('Number of samples')
plt.xlabel('Edibility')
plt.savefig('distribution_mushrooms.png')