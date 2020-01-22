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

from keras.models import load_model
#import personal files
import settings as st
import functions as fc
##########################################################################
#test later for logging
#import tensorboard
#import tempfile
#logdir = tempfile.mkdtemp()
#print('Writing training logs to ' + logdir)
# tensorboard - -logdir = {logdir}

##########################################################################

# find paths
train_directory = Path(__file__).parents[1].joinpath(st.rel_train_pathstring)
test_directory = Path(__file__).parents[1].joinpath(st.rel_test_pathstring)
poisonous_directory = Path(__file__).parents[1].joinpath(st.rel_poisonous_pathstring)

#set random seed
#import images and preprocess them
[train_image,train_image_labels] = fc.image_preprocessing(train_directory, st.NUM_CLASSES,
                                                          st.preprocessing_type)
[test_image, test_image_labels]= fc.image_preprocessing(test_directory, st.NUM_CLASSES,
                                                        st.preprocessing_type)
[poison_test_image, poison_test_image_labels] = fc.image_preprocessing(\
                                                    poisonous_directory,
                                                    st.NUM_POISON_TYPES,
                                                    st.preprocessing_type,
                                                    poison_identifier=True)
#show input
#plt.imshow(train_image[12, :, :, :])
#print(train_image_labels[12, :])
#print(train_image_labels.shape)

#initialize model
if st.training == True:
    our_model = fc.initialize_model(st.NUM_CLASSES)
    #get compiled model
    history = fc.compile_model(our_model, st.NUM_EPOCHS, train_image,
                            train_image_labels, test_image, test_image_labels)
    model_1 = history.model
    fc.plotting_Accuracy_Loss(st.NUM_EPOCHS, history, st.rel_pic_pathstring)
else:
    model_1 = load_model(Path(__file__).parents[1]\
                .joinpath(st.rel_model_load_pathstring))

if st.evaluation == True:
    results_clean = model_1.evaluate(test_image, test_image_labels)
    results_poison = model_1.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", results_clean)
    print("poison data test loss and testacc: ", results_poison)
    print("evaluation done")
    
if st.pruning == True:
    result_Prune = fc.pruning_channels(model_1, test_image, test_image_labels,
                                       st.DROP_ACC_RATE, 'conv2d_3')
    print(result_Prune[1])

#if plotting is set to True in settings: Plot accuracy Plot
if st.plotting == True:
    print("plot done")

# if saving is set to True in settings: save model
if st.saving ==True:
    fc.saving_model(model_1, st.rel_model_save_pathstring)
    print("saving model done")


