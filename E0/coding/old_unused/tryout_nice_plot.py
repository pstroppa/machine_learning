"""
.. module:: colored_attributes.py
    :platform:  Windows
    :synopsis: this file creates jitter plot for the wine_red.csv attributes

.. moduleauthor: Peter Stroppa
.. moduleeditor: Sophie Rain, Lucas Unterberger

"""
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.palettes import Viridis11
from bokeh.plotting import figure
from scipy.stats.kde import gaussian_kde
from bokeh.sampledata.perceptions import probly
import colorcet as cc

filename='data/wine_red.csv'

wine_dataframe = pd.read_csv(filename, sep =";", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
wine_dataframe_attributes = wine_dataframe.iloc[:,:-1].copy()

wine_dataframe_attributes2 = wine_dataframe[['fixed acidity', 'chlorides', 'volatile acidity', 'citric acid', 'free sulfur dioxide', 'pH', 'density', 'total sulfur dioxide', 'sulphates','residual sugar', 'alcohol']]
output_file("ridgeplot.html")

def ridge(category, data, scale=2.3):
    return list(zip([category]*len(data), scale*data))

cats = list(wine_dataframe_attributes2.keys())
palette = Viridis11

x = np.linspace(-20,70, wine_dataframe_attributes2.shape[0])
source = ColumnDataSource(data=dict(x=x))

p = figure(y_range=cats, plot_width=900, x_range=(-1, 15))

for i, cat in enumerate(reversed(cats)):
    pdf = gaussian_kde(wine_dataframe_attributes2[cat])
    y = ridge(cat, pdf(x))
    source.add(y, cat)
    p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)

p.outline_line_color = None
p.background_fill_color = "#efefef"

p.xaxis.ticker = FixedTicker(ticks=list(range(0, 40, 2)))
p.ygrid.grid_line_color = None
p.xgrid.grid_line_color = "#dddddd"
p.xgrid.ticker = p.xaxis[0].ticker

p.axis.minor_tick_line_color = None
p.axis.major_tick_line_color = None
p.axis.axis_line_color = None

p.y_range.range_padding = 0.5

show(p)
