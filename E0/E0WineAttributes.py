"""
.. module:: EOWineAttributs_colored.py
    :platform:  Windows
    :synopsis: this file creates jitter plot for the wine_red.csv attributes

.. moduleauthor: Peter Stroppa
.. moduleeditor: Sophie Rain, Lucas Unterberger

"""
#%%
import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Label, Range1d
from bokeh.palettes import Viridis11
from bokeh.plotting import figure, output_file, show
from bokeh.transform import jitter
from bokeh.layouts import gridplot
from math import pi

filename='wine_red.csv'

wine_dataframe = pd.read_csv("E2/data/" + filename, sep=";",
                            lineterminator="\n", encoding="utf-8", error_bad_lines=False)

wine_dataframe_attributes = wine_dataframe.iloc[:,:-1].copy()
attributes = wine_dataframe_attributes.columns.to_list()

figure_dict ={"figures":{},"sources":{}}
figure_dict["figures"] 
output_file("pic.html")
for i in range(wine_dataframe_attributes.shape[1]):
    data_plot = pd.DataFrame(wine_dataframe_attributes.iloc[:,i])
    data_plot["attribute_name"] = attributes[i]
    if attributes[i] == 'free sulfur dioxide' or attributes[i] == 'total sulfur dioxide':
        label = "mg/dm^3"
    elif attributes[i] == "density":
        label = "m/cm^3"
    elif attributes[i] == "alcohol":
        label = "vol. %"
    elif attributes[i] == "pH":
        label = "pH"
    else:
        label = 'g/dm^3'
    
    figure_dict["sources"][attributes[i]] = ColumnDataSource(data_plot)

    mapper = LinearColorMapper(palette=Viridis11, low=min(data_plot[attributes[i]]), high=max(data_plot[attributes[i]]))
    color_bar = ColorBar(color_mapper=mapper, location=(0,0))

    figure_dict["figures"][attributes[i]] = figure(y_range=[attributes[i]], title=attributes[i], x_axis_label=label , toolbar_location=None)
    opts = dict(x=attributes[i], line_color=None, source=figure_dict["sources"][attributes[i]])
    figure_dict["figures"][attributes[i]].circle(y=jitter("attribute_name", width=0.9, range=figure_dict["figures"][attributes[i]].y_range), fill_color={'field': attributes[i], 'transform': mapper}, **opts)
    figure_dict["figures"][attributes[i]].add_layout(color_bar, 'right')

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

graphic = gridplot([figure_dict["figures"][attributes[i]] for i in range(wine_dataframe_attributes.shape[1])], ncols=4, plot_width=300, plot_height=300)
show(graphic)
#export_png(graphic, filename="attributes_winequality_bunt.png")
#%%
output_file("sugar_quality.html")
source = ColumnDataSource(data=dict(x=wine_dataframe["total sulfur dioxide"], y=wine_dataframe["quality"]))
p = figure(plot_height=850, plot_width=850, x_axis_label=('sugar'), y_axis_label=('quality'))
p.circle(x="x", y="y", line_width=2, source=source)
show(p)
# %%
