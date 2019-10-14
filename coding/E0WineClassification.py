"""
.. module:: E0WineClassification.py
    :platform:  Windows
    :synopsis: this file creates histograms for the wine_red.csv attribute and classifications

.. moduleauthor: Sophie Rain
.. moduleeditor: Peter Stroppa, Lucas Unterberger

"""

#imported for compatibility of python 2.x and 3.x
from __future__ import unicode_literals
from future.builtins import map  
#imports Counter class
from collections import Counter
#imports chain generator objects
from itertools import chain
#used for data manipulation and DataFrame datatype
import pandas as pd
#bokeh is used for nice plotting
from bokeh.io import export_png
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.palettes import Spectral11


filename='data/wine_red.csv'

wine_dataframe = pd.read_csv(filename, sep =";")
samples = wine_dataframe["quality"] 
classes = [str(i).zfill(1) for i in range(0,11)]
values=[0]*11

for elem in samples:
        values[int(elem)]=values[int(elem)]+1

source = ColumnDataSource(data=dict(classes=classes, counts=values, color=Spectral11))
p = figure(x_range=classes, plot_height=500, title="Distribution Wine Quality", x_axis_label=('Quality Value'),y_axis_label=('Number of samples'), toolbar_location=None)
p.vbar(x='classes', top='counts', width=0.9, color='color', source=source) #legend="classes"

p.xgrid.grid_line_color = "white"
p.y_range.start = 0

p.title_location = "above"
p.title.align = 'center'
p.title.text_font_size = '16pt'

p.xaxis.axis_label_text_font_size = "16pt"
p.yaxis.axis_label_text_font_size = "16pt" 
p.xaxis.major_label_text_font_size = "14pt"

#p.legend.label_text_font_size = "16pt"
p.y_range=Range1d(0,850)

#p.legend.location = "top_left"
#p.legend.orientation = "horizontal"
#p.legend.location = "top_center"

for i in range(11):
    if i < 3:
        p.add_layout(Label(x=20+49*i, y=int(values[i]), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))   
    elif i == 9 or i ==10:
        p.add_layout(Label(x=17+49*i, y=int(values[i]), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))   
    elif i==4:
        p.add_layout(Label(x=15+49*i, y=int(values[i]/2), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))
    elif values[i]<20:
        p.add_layout(Label(x=13+49*i, y=int(values[i]-8), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))
    elif values[i]<100:
        p.add_layout(Label(x=10+49*i, y=int(values[i]/2), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))
    elif values[i]<200:
        p.add_layout(Label(x=9+49*i, y=int(values[i]/2.09), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))
    else:
        p.add_layout(Label(x=10+49*i, y=int(values[i]/2.09), x_units='screen', y_units='screen', text=str(values[i]), text_color = "black"))

export_png(p, filename="pics/distribution_winequality.png")