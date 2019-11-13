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

bike_df = pd.read_csv("E1/data/bikeSharing_train.csv", sep =",",                                                                          lineterminator="\n", encoding="utf-8",error_bad_lines=False)
bike_test_df = pd.read_csv("E1/data/bikeSharing_test.csv", sep =",", lineterminator="\n",                                                      encoding="utf-8", error_bad_lines=False)

bike_df = bike_df.set_index("id", drop=True)
bike_df
bike_test_df = bike_test_df.set_index("id", drop=True)
#bike_train_df = bike_train_df.sort_values(by="id")
################################################################

bike_train_attributes=bike_train_df.drop(columns=["cnt""dteday"])
bike_test_attributes=bike_test_df.drop(columns=["dteday"])