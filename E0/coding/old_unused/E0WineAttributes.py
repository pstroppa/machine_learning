"""
.. module:: E0WineAttributes.py
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
#%%
filename='data/wine_red.csv'

wine_dataframe = pd.read_csv(filename, sep =";", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
wine_dataframe_attributes = wine_dataframe.iloc[:,:-1].copy()
attributes = wine_dataframe_attributes.columns.to_list()

#%%
figure_dict ={"figures":{},"sources":{}}
figure_dict["figures"] 
for i in range(wine_dataframe_attributes.shape[1]):
    data_plot = pd.DataFrame(wine_dataframe_attributes.iloc[:,i])
    data_plot["attribute_name"] = attributes[i]
    if attributes[i] == 'free sulfur dioxide' or attributes[i] == 'total sulfur dioxide':
        label = "mg/gm^3"
    elif attributes[i] == "density":
        label = "m/cm^3"
    elif attributes[i] == "alcohol":
        label = "vol. %"
    elif attributes[i] == "pH":
        label = "pH"
    else:
        label = 'g/dm^3'
    
    figure_dict["sources"][attributes[i]] = ColumnDataSource(data_plot)

    figure_dict["figures"][attributes[i]] = figure(y_range=[attributes[i]], title=attributes[i], x_axis_label=label , toolbar_location=None)

    figure_dict["figures"][attributes[i]].circle(x=attributes[i], y=jitter("attribute_name", width=0.9, range=figure_dict["figures"][attributes[i]].y_range), size=3, source=figure_dict["sources"][attributes[i]], alpha=0.3)

    figure_dict["figures"][attributes[i]].xgrid.grid_line_color = "white"

    figure_dict["figures"][attributes[i]].title_location = "above"
    figure_dict["figures"][attributes[i]].title.align = 'center'
    figure_dict["figures"][attributes[i]].title.text_font_size = '16pt'
    figure_dict["figures"][attributes[i]].xaxis.major_label_orientation = pi/2
    figure_dict["figures"][attributes[i]].yaxis.major_label_orientation = pi/2

    figure_dict["figures"][attributes[i]].xaxis.axis_label_text_font_size = "16pt"
    figure_dict["figures"][attributes[i]].yaxis.axis_label_text_font_size = "16pt"
    figure_dict["figures"][attributes[i]].xaxis.major_label_text_font_size = "14pt"
    figure_dict["figures"][attributes[i]].yaxis.major_label_text_font_size = "14pt"
    figure_dict["figures"][attributes[i]].yaxis.major_label_text_align = "left"
    figure_dict["figures"][attributes[i]].yaxis.visible = False
   
    if max(data_plot[attributes[i]]) - min(data_plot[attributes[i]]) < 0.3:
        figure_dict["figures"][attributes[i]].x_range = Range1d(min(data_plot[attributes[i]])-0.0001, max(data_plot[attributes[i]])+0.0001)
    elif max(data_plot[attributes[i]]) - min(data_plot[attributes[i]]) < 1:
        figure_dict["figures"][attributes[i]].x_range = Range1d(min(data_plot[attributes[i]])-0.01, max(data_plot[attributes[i]])+0.01)
    elif max(data_plot[attributes[i]]) - min(data_plot[attributes[i]]) < 5:
        figure_dict["figures"][attributes[i]].x_range = Range1d(min(data_plot[attributes[i]])-0.3, max(data_plot[attributes[i]])+0.3)
    else: 
        figure_dict["figures"][attributes[i]].x_range = Range1d(min(data_plot[attributes[i]])-0.5, max(data_plot[attributes[i]])+0.5)


graphic = gridplot([figure_dict["figures"][attributes[i]] for i in range(wine_dataframe_attributes.shape[1])], ncols=4, plot_width=300, plot_height=300, toolbar_location=None)
export_png(graphic, filename="pics/attributes_winequality.png")




