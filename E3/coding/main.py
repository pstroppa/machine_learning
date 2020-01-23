'''
.. module:: main.py
    :platform:   Windows
    :synopsis:   this main file is used for running the porject
                 Machine Learning E3 group 20 (Task 3.1.3) it
                 imports: settings.py and functions.py

.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''

# import path and numpy
from pathlib import Path

# import libraries for plotting and calculation
import matplotlib.pyplot as plt

from keras.models import load_model

#temp import numpy
import numpy as np
import pandas as pd
# import personal files
import settings as st
import functions as fc
##########################################################################
# test later for logging
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
# set random seed
np.random.seed=(st.seed)
# import images and preprocess them
[train_image, train_image_labels] = fc.image_preprocessing(train_directory, st.NUM_CLASSES,
                                                           st.preprocessing_type)
[test_image, test_image_labels] = fc.image_preprocessing(test_directory, st.NUM_CLASSES,
                                                         st.preprocessing_type)
[poison_test_image, poison_test_image_labels] = fc.image_preprocessing(
    poisonous_directory,
    st.NUM_POISON_TYPES,
    st.preprocessing_type,
    poison_identifier=True)
# show input
#plt.imshow(train_image[12, :, :, :])
#print(train_image_labels[12, :])
# print(train_image_labels.shape)
# initialize model

if st.standard_attack == True:
    if st.training == True:
        load_path_standard = None
    else:
        load_path_standard = st.rel_model_load_pathstring
    standard_model = fc.standard_attack(st.NUM_CLASSES, st.preprocessing_type, st.NUM_EPOCHS,
                                        train_image, train_image_labels, test_image,
                                        test_image_labels, poison_test_image, poison_test_image_labels, st.rel_pic_pathstring, load_path_standard)


# attacke uses an pruning aware attack e.g. following 4 Steps
if st.pruning_aware_attack == True:
    if st.pruning_aware_training == True:
        load_path = None
    else:
        load_path = st.rel_clean_model_load_pathstring

    values = pd.DataFrame(columns=[
                          "drop_rate", "epochs", "learning", "ratio", "bias", "clean_acc", "pois_acc", "score"])
    ratios = [0.05, 0.1, 0, 2]
    count = 0
    for drop_rate in np.arange(0.99, 0.999, 0.003):
        for epochs in np.arange(80, 120, 10):
            for learning in np.arange(0.0005, 0.01, 0.004):
                for ratio in ratios:
                    for bias in np.arange(0.4, 0.6, 0.1):
                        paa_model = fc.pruning_aware_attack(train_directory, st.preprocessing_type, st.NUM_CLASSES, st.NUM_EPOCHS,
                                                            test_image, test_image_labels, poison_test_image, poison_test_image_labels,
                                                            drop_rate, st.layer_name, epochs, learning, ratio, bias, load_path)
                        clean_acc = paa_model.evaluate(
                            test_image, test_image_labels)
                        pois_acc = paa_model.evaluate(
                            poison_test_image, poison_test_image_labels)
                        score = clean_acc[1]*0.6+pois_acc[1]*0.4
                        df2 = pd.DataFrame([[drop_rate, epochs, learning, ratio, bias, clean_acc[1], pois_acc[1], score]],
                                           columns=["drop_rate", "epochs", "learning", "ratio", "bias", "clean_acc", "pois_acc", "score"])
                        values = values.append(df2)
                        count += 1
                        print(count)
    values.to_csv('E3/coding/parameter_paa.csv', sep=';', encoding='utf-8')

    #paa_model = pruning_aware_attack(st.train_directory, st.preprocessing_type, st.NUM_CLASSES, st.NUM_EPOCHS,
    #                                 test_image, test_image_labels, poison_test_image, poison_test_image_labels,
    #                                 st.DROP_ACC_RATE_PAA,
    #                                 st.layer_name, st.n_epochs_paa, st.learning_rate_paa, st.train_test_ratio_paa,
    #                                 st.bias_decrease, load_path)


# evaluate accuracy for clean and poisonous data
if st.evaluation == True:
    print('evaluation done')
# prune model in layer "conv2d_3" while accuracy does not drop bellow DROP_ACC_RATE%
if st.pruning == True:
    pruned_model, accuracy_pruned, number_nodes_pruned = fc.pruning_channels(standard_model,
                                                                             test_image,
                                                                             test_image_labels,
                                                                             st.DROP_ACC_RATE, 'conv2d_3')
    print('accuracy', accuracy_pruned)
    backdoor = pruned_model.evaluate(poison_test_image, poison_test_image_labels)
    backdoor_success = backdoor[1]
    print('backdoor_success', backdoor_success)
    print("pruning done")

# fine tune model with specified learning rate and number of epochs
if st.fine_tuning == True:
    if st.pruning == True:
        # fine pruning. First prune and then on top of it fine tune
        fine_tuned_history = fc.fine_tuning_model(pruned_model, st.fine_tuning_n_epochs,
                                                  st.fine_tuning_learning_rate, test_image,
                                                  test_image_labels, st.fine_tuning_ratio)
        print("fine pruning done")
    else:
        fine_tuned_history = fc.fine_tuning_model(model_1, st.fine_tuning_n_epochs,
                                                  st.fine_tuning_learning_rate, test_image,
                                                  test_image_labels, st.fine_tuning_ratio)
        print("fine tuning done")

# if plotting is set to True in settings: Plot accuracy Plot
if st.plotting == True:
    print("plot done")

# if saving is set to True in settings: save model
if st.saving == True:
    fc.saving_model(model_1, st.rel_model_save_pathstring)
    print("saving standard model done")

if st.paa_save == True:
    fc.saving_model(paa_model, st.rel_model_paa_save_pathstring)
    print("saving paa model done")
