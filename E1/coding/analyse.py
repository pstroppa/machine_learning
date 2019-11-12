"""
.. module:: E1linearfit.py
    :platform:  Windows
    :synopsis: creates Linear Fit, for several datasets

.. moduleauthor: Peter Stroppa
.. moduleeditor: Sophie Rain, Lucas Unterberger

"""
import numpy as np
from math import pi
import datetime as dt
import pandas as pd
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Label, Range1d
from bokeh.palettes import Viridis11
from bokeh.plotting import figure
from bokeh.transform import jitter
from bokeh.layouts import gridplot
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

bike_train_df = pd.read_csv("E1/data/bikeSharing_train.csv", sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
bike_test_df = pd.read_csv("E1/data/bikeSharing_test.csv", sep =",", lineterminator="\n", encoding="utf-8",error_bad_lines=False)

bike_train_df = bike_train_df.set_index("id", drop=True)
bike_train_df = bike_train_df.sort_values(by="id")
################################################################
## analyse DataFrame for possible preprocessing

# possible outlier:
#weathersit: 26.1.2011  16h
 #           10.1.2012 18h


#atemp:       22.07.2011 14h
#             17.8.2012  0-18h
#             04.01.2011 4h
#             12.02.2011 4h

#hum:         10.03.2011

#windspeed    03.07.2011 17h
#             27.08.2011 17h

grouped = {}
for col in bike_train_df.columns.to_list():
    grouped[col]=bike_train_df[col].groupby(bike_train_df[col]).count()

attributes = ["date", "weathersit", "temp", "atemp", "hum", "windspeed"]

for attribute in attributes:    
    source = ColumnDataSource(data=dict(x=bike_train_df[attribute], y=bike_train_df["cnt"]))
    output_file(attribute +".html")
    p = figure(plot_width=900, plot_height=900, title=attribute, x_axis_label=('Date'),y_axis_label=('Value'))#, toolbar_location=None) x_axis_type="datetime"
    p.circle(x="x", y="y",  source=source, alpha=0.3)
    p.xgrid.grid_line_color = "white"
    p.title_location = "above"
    p.title.align = 'center'
    p.title.text_font_size = '16pt'
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt" 
    p.xaxis.major_label_text_font_size = "14pt"
    show(p)
####################################################################
#Ziel: plot attribute vs count. (attribute sort)

#explantation what is happening:
#random x values
x=np.random.randn(10,1)
#y is the funktion 2*x+3 + some random noise
y=2*x+3 +0.5*np.random.rand(10,1)
#fit modell so that it finds a linear function that suits y closest possible
regr.fit(x,y)
#coef_ should ideally return 2 since we are looking for the closest function to y -> which is of course y
regr.coef_
# incerept is d, where y=k*x+d -> hence it should be 3
regr.intercept_
# now i want to know what the value for 20 would be for a our approximated function
regr.predict([[20]])
x_test = np.linspace(-3,3) 

####################################################################

plt.scatter(bike_train_df["temp"], bike_train_df["cnt"],  color='white')
plt.plot(test_attribute, test_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()