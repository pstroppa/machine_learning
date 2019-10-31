"""
.. module:: old_wine_attributes.py
    :platform:  Windows
    :synopsis: this file creates jitter plot for the wine_red.csv attributes

.. moduleauthor: Peter Stroppa
.. moduleeditor: Sophie Rain, Lucas Unterberger

"""
#imports Counter class
from itertools import chain
#used for data manipulation and DataFrame datatype
import pandas as pd
from math import pi
#bokeh is used for nice plotting
from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.palettes import Viridis11
from bokeh.plotting import figure
from bokeh.io import export_png, show, output_file
from bokeh.transform import jitter
from bokeh.layouts import gridplot

filename='data/wine_red.csv'

wine_dataframe = pd.read_csv(filename, sep =";", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
wine_dataframe_attributes = wine_dataframe.iloc[:,:-1].copy()
attributes = wine_dataframe_attributes.columns.to_list()


# wine_dataframe_attributes = wine_dataframe.iloc[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'sulphates', 'alcohol']].copy()
flat = pd.DataFrame(wine_dataframe_attributes.iloc[0])
flat.columns={"values"}
for instance,attribute in wine_dataframe_attributes.iloc[1:].iterrows():
    temp = pd.DataFrame(wine_dataframe_attributes.loc[instance])
    temp.columns={"values"}
    flat = flat.append(temp)


source = ColumnDataSource(data=dict(attributes=flat.index.to_list(), values=flat["values"]))

output_file("bars.html")

p = figure(plot_width=900, plot_height=900, y_range=attributes, title="distribution of wine-quality attributes", x_axis_label=('g/dm^3'),y_axis_label=('attributes'))#, toolbar_location=None)

p.circle(x="values", y=jitter("attributes", width=1.5, range=p.y_range),  source=source, alpha=0.3)

p.xgrid.grid_line_color = "white"

p.title_location = "above"
p.title.align = 'center'
p.title.text_font_size = '16pt'

p.xaxis.axis_label_text_font_size = "16pt"
p.yaxis.axis_label_text_font_size = "16pt" 
p.xaxis.major_label_text_font_size = "14pt"
p.x_range=Range1d(-0.5,15)

show(p)
