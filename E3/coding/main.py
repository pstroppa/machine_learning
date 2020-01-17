'''
.. module:: main.py
    :platform:   Windows
    :synopsis:   this main file is used for running the porject
                 Machine Learning E3 group 20 (Task 3.1.3) it
                 imports: settings.py and functions.py

.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''

#import path and numpy
from pathlib import Path
import numpy as np

#import libraries for plotting and calculation
import matplotlib.pyplot as plt

#import personal files
import settings as st
import functions as fc
##########################################################################


# find paths
traindirectory = Path(__file__).parents[1].joinpath(st.rel_train_pathstring)
testdirectory = Path(__file__).parents[1].joinpath(st.rel_test_pathstring)

#set random seed
np.random.seed(st.seed)

#import images and preprocess them
[train_image,train_image_labels] = fc.image_preprocessing(traindirectory, st.NUM_CLASSES,
                                                          st.preprocessing_type)
[test_image, test_image_labels]= fc.image_preprocessing(testdirectory, st.NUM_CLASSES,
                                                        st.preprocessing_type)

#show input
plt.imshow(train_image[12, :, :, :].reshape(32,32), cmap='gray')
#print(train_image_labels[12, :])
#print(train_image_labels.shape)

#initialize model
our_model = fc.initialize_model(st.NUM_CLASSES)

#get compiled model
history = fc.compile_model(our_model, st.NUM_EPOCHS, train_image,
                        train_image_labels, test_image, test_image_labels)

#history_pruned = pruning_model(history)

#if plotting is set to True in settings: Plot accuracy Plot
if st.plotting == True:
    fc.plotting_Accuracy_Loss(st.NUM_EPOCHS, history, st.rel_pic_pathstring)
    print("plot done")

# if saving is set to True in settings: save model
if st.saving ==True:
    fc.saving_model(our_model, st.rel_model_pathstring)
    print("saving done")
