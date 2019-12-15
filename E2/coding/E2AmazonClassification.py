"""
.. module:: E0dataClassification.py
    :platform:  Windows
    :synopsis: this file creates histograms for the amazon_red.csv attribute and classifications

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
from bokeh.plotting import figure, output_file, show

from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.palettes import magma
from math import pi


filename='E2/data/amazon_train.csv'

amazon_dataframe = pd.read_csv(filename, sep =",")
samples = amazon_dataframe["Class"] 
grouped = samples.groupby(samples).count()
classes = grouped.index.to_list()
values = grouped.values.tolist()
source = ColumnDataSource(data=dict(classes=classes,
                                    counts=values,
                                    color=magma(50)))
p = figure(x_range=classes, plot_height=600, plot_width=1000, title="Distribution Amazon Authors",
           x_axis_label=('Authors'), y_axis_label=('Number of samples'), toolbar_location=None)
p.vbar(x='classes', top='counts', width=0.8, color="color", source=source) #legend="classes"

p.xgrid.grid_line_color = "white"
p.y_range.start = 0

p.title_location = "above"
p.title.align = 'center'
p.title.text_font_size = '18pt'

p.xaxis.axis_label_text_font_size = "16pt"
p.yaxis.axis_label_text_font_size = "16pt" 
p.xaxis.major_label_text_font_size = "12pt"
p.xaxis.major_label_orientation = pi/2

#p.legend.label_text_font_size = "16pt"
p.y_range=Range1d(0,25)

#p.legend.location = "top_left"
#p.legend.orientation = "horizontal"
#p.legend.location = "top_center"

#output_file("pic.html")

#show(p)
export_png(p, filename="pics/distribution_amazon.png")

