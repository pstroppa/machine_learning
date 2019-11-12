"""
.. module:: E1linearfit.py
    :platform:  Windows
    :synopsis: creates Linear Fit, for several datasets

.. moduleauthor: Peter Stroppa
.. moduleeditor: Sophie Rain, Lucas Unterberger

"""
#%%
#import standart packages
import numpy as np
from math import pi
import datetime as dt
import pandas as pd
#import plotting packages
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.plotting import figure, output_file, show
from bokeh.transform import jitter
#import machine learning packages
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

bike_train_df = pd.read_csv("E1/data/bikeSharing_train.csv", sep =",",                                                                          lineterminator="\n", encoding="utf-8",error_bad_lines=False)
bike_test_df = pd.read_csv("E1/data/bikeSharing_test.csv", sep =",", lineterminator="\n",                                                      encoding="utf-8", error_bad_lines=False)

bike_train_df = bike_train_df.set_index("id", drop=True)
bike_test_df = bike_test_df.set_index("id", drop=True)
#bike_train_df = bike_train_df.sort_values(by="id")
################################################################

split = bike_train_df["dteday"].iloc[0].split("-") 
loc = [dt.datetime(year=int(split[0]),month=int(split[1]),day=int(split[2]), 
                   hour=int(bike_train_df["hr"].iloc[0]))]
for i in range(1,bike_train_df.shape[0]):
    split = bike_train_df["dteday"].iloc[i].split("-") 
    loc.append(dt.datetime(year=int(split[0]),month=int(split[1]),day=int(split[2]),
                           hour=int(bike_train_df["hr"].iloc[i])))
bike_train_df["date"]=loc

bike_train_attributes=bike_train_df.drop(columns=["cnt", "date", "dteday"])
bike_test_attributes=bike_test_df.drop(columns=["dteday"])

#attributes = ["date", "weathersit", "temp", "atemp", "hum", "windspeed"]
################################################################
# machine lerarning part
####################################################################

corr=bike_train_df.corr()
corr["cnt"]
value = bike_train_df["cnt"].values[:,np.newaxis]
# Use only one feature
train_attribute = bike_train_attributes["temp"].values[:,np.newaxis]
#train_attribute=bike_train_attributes.values
test_attribute= bike_test_attributes["temp"].values[:,np.newaxis]
#test_attribute=bike_test_attributes.values

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_attribute, value)
# Make predictions using the testing set
#train_pred = regr.predict(train_attribute)
test_pred = regr.predict(test_attribute)

regr.score(train_attribute[:-1],test_pred)

flat_test_attribute = [item for sublist in test_attribute for item in sublist]
flat_prediction = [item for sublist in test_attribute for item in sublist]

# Plot outputs
source = ColumnDataSource(data=dict(x=bike_train_df["temp"], y=bike_train_df["cnt"]))
output_file("temp" +".html")
p = figure(plot_width=800, plot_height=800, title="date-jitter", x_axis_label=('Temp-normalized'),
           y_axis_label=('Value'))#, toolbar_location=None)
p.circle(x="x", y="y", source=source, alpha=0.3)
p.line(x=flat_test_attribute, y=[x * np.asscalar(regr.coef_) + np.asscalar(regr.intercept_)
 for x in flat_test_attribute], color='blue', line_width=3)
p.xgrid.grid_line_color = "white"
p.title_location = "above"
p.title.align = 'center'
p.title.text_font_size = '16pt'
p.xaxis.axis_label_text_font_size = "16pt"
p.yaxis.axis_label_text_font_size = "16pt" 
p.xaxis.major_label_text_font_size = "14pt"
show(p)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(value, test_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(value, test_pred))




# %%
