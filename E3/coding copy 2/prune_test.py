#import libraries for paths
from pathlib import Path
import glob
import numpy as np
#import libraries for preprocessing pictures
import cv2
from skimage import color, exposure, transform

#import libraries for plotting and calculation
import matplotlib.pyplot as plt

#import librariers for CNN and Deep Learning (Tensorflow in Backend)
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

#libraries for pruning
from kerassurgeon import identify, surgeon
from kerassurgeon.operations import delete_channels, delete_layer
import pandas as pd

model = Sequential()
input_shape = (32, 32, 1)  # images of 32x32 and 3 layers (rgb)

model.add(Conv2D(32, (5, 5), padding='same',
                 activation='relu', input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(9, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])


model.load_weights(Path(__file__).parents[0].joinpath("weights_grey.h5"))


lay6 = model.layers[0]

surgeon = surgeon.Surgeon(model)
surgeon.add_job('delete_channels', lay6, channels=[5,8,26])
new_model = surgeon.operate()

new_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

