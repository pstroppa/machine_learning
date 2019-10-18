"""
.. module:: E0MushroomClassification.py
    :platform:  Windows
    :synopsis: this file creates histograms for the mushrooms.csv attribute and classifications

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
from bokeh.palettes import Spectral6

filename='data/mushroom.csv'
classes = ['edible','poisonous']

mushroom_dataframe = pd.read_csv(filename, sep =",")
samples = mushroom_dataframe["p"]  

values=[0,0]

for elem in samples:
    if elem[0]=='e':
        values[0]=values[0]+1
    else:
        values[1]=values[1]+1

source = ColumnDataSource(data=dict(classes=classes, counts=values, color=Spectral6[0:2]))
p = figure(x_range=classes, plot_height=500, title="Distribution Mushrooms", x_axis_label=('Edibility'),y_axis_label=('Number of samples'), toolbar_location=None)
p.vbar(x='classes', top='counts', width=0.9, color='color', source=source) #legend ="classes"

p.xgrid.grid_line_color = "white"
p.y_range.start = 0

p.title_location = "above"
p.title.align = 'center'
p.title.text_font_size = '20pt'

p.xaxis.axis_label_text_font_size = "20pt"
p.yaxis.axis_label_text_font_size = "20pt" 
p.xaxis.major_label_text_font_size = "16pt"


#p.legend.label_text_font_size = "16pt"
p.y_range=Range1d(0,5000)

#p.legend.location = "top_left"
#p.legend.orientation = "horizontal"
#p.legend.location = "top_center"

citation1 = Label(x=100, y=285, x_units='screen', y_units='screen', text=str(values[0]), text_color = "white", text_font_size = "20pt")
citation2 = Label(x=360, y=265, x_units='screen', y_units='screen', text=str(values[1]), text_color = "white", text_font_size = "20pt")

p.add_layout(citation1)
p.add_layout(citation2)

export_png(p, filename="data/distribution_mushrooms.png")






