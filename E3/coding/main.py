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
[clean_train_image, clean_train_image_labels] = fc.image_preprocessing(train_directory, st.NUM_CLASSES,
                                                                       st.preprocessing_type, train_add_poison=False)

[test_image, test_image_labels] = fc.image_preprocessing(test_directory, st.NUM_CLASSES,
                                                         st.preprocessing_type)
[poison_test_image, poison_test_image_labels] = fc.image_preprocessing(poisonous_directory,
                                                                       st.NUM_POISON_TYPES,
                                                                       st.preprocessing_type,
                                                                       poison_identifier=True)
# show input
#plt.imshow(train_image[12, :, :, :])
#print(train_image_labels[12, :])
# print(train_image_labels.shape)
# initialize model

if st.standard_attack == True:
    if st.standard_training == True:
        load_path_standard = None
    else:
        load_path_standard = st.rel_model_load_pathstring
    
    standard_model = fc.standard_attack(st.NUM_CLASSES, st.preprocessing_type, st.NUM_EPOCHS,
                                        train_image, train_image_labels, test_image,
                                        test_image_labels, poison_test_image, poison_test_image_labels, st.rel_pic_pathstring,
                                        st.rel_model_save_pathstring, load_path_standard)
    
    #make a "deep" copy of the model i.e. save and reload the model
    standard_model_for_plotting = fc.clone_model(standard_model,"plot")
    
    #if prune_plot true , plot accuracy and backdoor success, given the number of pruned nodes
    if st.prune_plot == True:
        y_clean, y_pois = fc.pruning_for_plot(standard_model_for_plotting, test_image,
                                            test_image_labels, poison_test_image, poison_test_image_labels,
                                            'conv2d_3')

        fc.plot_pruned_neurons_clean_and_backdoor_accuracy(y_clean, y_pois, 'prune_plot_standard')
        print("created accuracy loss when all nodes pruned plot")
    # evaluate accuracy for clean and poisonous data
    if st.standard_evaluate_defenses == True:

        values=[]
        #no defense
        no_def_clean = standard_model.evaluate(test_image, test_image_labels)
        no_def_poison = standard_model.evaluate(poison_test_image, poison_test_image_labels)

        values.append([no_def_clean[1], no_def_poison[1]])

        #make a "deep" copy of the model i.e. save and reload the model
        standard_model_for_pruning1 = fc.clone_model(standard_model,"prune")
        #pruning
        pruned_model, accuracy_pruned, number_nodes_pruned, indices_ignore = fc.pruning_channels(standard_model_for_pruning,
                                                                                test_image,
                                                                                test_image_labels,
                                                                                st.DROP_ACC_RATE, 'conv2d_3')
        fc.saving_model(pruned_model, 'models/standard_pruned_' + st.model_name + '.h5')

        backdoor = pruned_model.evaluate(poison_test_image, poison_test_image_labels)
        backdoor_success = backdoor[1]

        print('pruning done and model saved')
        values.append([accuracy_pruned, backdoor_success])

        #fine_tuning
        fine_tuned_history = fc.fine_tuning_model(standard_model, st.fine_tuning_n_epochs,
                                                  st.fine_tuning_learning_rate, clean_train_image,
                                                  clean_train_image_labels, st.fine_tuning_ratio, st.seed)
        fine_tuned_model=fine_tuned_history.model
        fine_tuned_clean = fine_tuned_model.evaluate(test_image, test_image_labels)
        fine_tuned_poison = fine_tuned_model.evaluate(poison_test_image, poison_test_image_labels)

        print('clean_acc_fine_tuned', fine_tuned_clean[1])
        print('backdoor_success_fine_tuned', fine_tuned_poison[1])
        values.append([fine_tuned_clean[1], fine_tuned_poison[1]])
        fc.saving_model(fine_tuned_model, 'models/standard_fine_tuned_' + st.model_name + '.h5')

        #fine_pruning
        fine_pruned_history = fc.fine_tuning_model(pruned_model, st.fine_tuning_n_epochs,
                                                   st.fine_tuning_learning_rate, clean_train_image,
                                                   clean_train_image_labels, st.fine_tuning_ratio, st.seed)

        fine_pruned_model = fine_pruned_history.model
        standard_model.evaluate(test_image, test_image_labels)
        fine_pruned_clean = fine_pruned_model.evaluate(test_image, test_image_labels)
        fine_pruned_poison = fine_pruned_model.evaluate(poison_test_image, poison_test_image_labels)

        print('clean_acc_fine_pruned', fine_pruned_clean[1])
        print('backdoor_success_fine_pruned', fine_pruned_poison[1])
        values.append([fine_pruned_clean[1], fine_pruned_poison[1]])
        fc.saving_model(fine_pruned_model, 'models/standard_fine_pruned_' + st.model_name + '.h5')
        
        defense_accuracy=pd.DataFrame(values, index=['no_defense', 'pruning','fine_tuning',
                                      'fine_pruning'], columns=['clean','backdoor'])

        defense_accuracy.to_csv(Path(__file__).parents[1].joinpath('results/defense_accuracy_standard_'/
                                                                   + st.model_name + '.csv'), sep=";", line_terminator="\n", encoding="utf-8")
        print('evaluation done')


# attacke uses an pruning aware attack e.g. following 4 Steps
if st.pruning_aware_attack == True:
    #if load only is flase, do steps 2-4 of paa, if true load paa model
    if st.paa_load_only == False:
        #if pruning_aware_training is true do step 1 (training of clean model)
        if st.pruning_aware_training == True:
            load_path = None
        else:
            #load new paa_model
            load_path = st.rel_clean_model_load_pathstring

        paa_model = fc.pruning_aware_attack(train_directory, st.preprocessing_type, st.NUM_CLASSES, st.NUM_EPOCHS,
                                        test_image, test_image_labels, poison_test_image, poison_test_image_labels,
                                        st.num_del_nodes_paa, st.layer_name, st.n_epochs_paa, st.learning_rate_paa,
                                        st.train_test_ratio_paa, st.bias_decrease, st.rel_paa_model_save_pathstring,
                                        st.seed, st.rel_clean_save_pathstring, load_path)
    else:
        paa_model = load_model(Path(__file__).parents[1]
                               .joinpath(st.rel_paa_model_load_pathstring))
    
    #if prune_plot_paa is true, then plot the accuracy and backdoor success, given number of pruned nodes
    if st.prune_plot_paa == True:
        print("start pruning plot for paa")
        #copy modell
        paa_model_for_plotting = fc.clone_model(paa_model,"paa_plot")

        y_clean, y_pois = fc.pruning_for_plot_paa(paa_model_for_plotting, test_image,
                                            test_image_labels, poison_test_image,
                                            poison_test_image_labels, 'conv2d_3')

        fc.plot_pruned_neurons_clean_and_backdoor_accuracy(y_clean, y_pois, 'prune_plot_paa_'+st.model_name)

    if st.paa_evaluate_defenses == True:
        values=[]
        #no defense
        no_def_clean = paa_model.evaluate(test_image, test_image_labels)
        no_def_poison = paa_model.evaluate(poison_test_image, poison_test_image_labels)

        print('no_def_clean', no_def_clean[1])
        print('no_def_backdoor_success', no_def_poison[1])
        values.append([no_def_clean[1], no_def_poison[1]])

        #copy modell
        paa_model_for_pruning = fc.clone_model(paa_model,"paa_prune")
        print("--------------------------------")
        print("Start pruning defense for paa")
        print("--------------------------------")
        #pruning
        pruned_model, accuracy_pruned, number_nodes_pruned, indices_ignore = fc.pruning_channels_paa(paa_model_for_pruning,
                                                                                test_image,
                                                                                test_image_labels,
                                                                                st.DROP_ACC_RATE, 'conv2d_3')

        print('clean_accuracy_pruned', accuracy_pruned)
        backdoor = pruned_model.evaluate(poison_test_image, poison_test_image_labels)
        backdoor_success = backdoor[1]
        print('backdoor_success_pruned', backdoor_success)
        values.append([accuracy_pruned, backdoor_success])
        fc.saving_model(pruned_model, 'models/paa_pruned_' + st.model_name + '.h5')
 
        ###############################################################################
        print("--------------------------------")
        print("Start fine tuning for paa")
        print("--------------------------------")
        #fine_tuning
        fine_tuned_history = fc.fine_tuning_model(paa_model, st.fine_tuning_n_epochs,
                                                  st.fine_tuning_learning_rate, clean_train_image,
                                                  clean_train_image_labels, st.fine_tuning_ratio, st.seed)
        fine_tuned_model = fine_tuned_history.model
        fine_tuned_clean = fine_tuned_model.evaluate(test_image, test_image_labels)
        fine_tuned_poison = fine_tuned_model.evaluate(poison_test_image, poison_test_image_labels)

        print('clean_acc_fine_tuned', fine_tuned_clean[1])
        print('backdoor_success_fine_tuned', fine_tuned_poison[1])
        values.append([fine_tuned_clean[1],fine_tuned_poison[1]])
        fc.saving_model(fine_tuned_model, 'models/paa_fine_tuned_' + st.model_name + '.h5')
        
        print("--------------------------------")
        print("Start fine pruning for paa")
        print("--------------------------------")
        #fine_pruning
        fine_pruned_history = fc.fine_tuning_model(pruned_model, st.fine_tuning_n_epochs,
                                                   st.fine_tuning_learning_rate, clean_train_image,
                                                   clean_train_image_labels, st.fine_tuning_ratio, st.seed)
        fine_pruned_model = fine_pruned_history.model
        fine_pruned_clean = fine_pruned_model.evaluate(test_image, test_image_labels)
        fine_pruned_poison = fine_pruned_model.evaluate(poison_test_image, poison_test_image_labels)

        print('clean_acc_fine_pruned', fine_pruned_clean[1])
        print('backdoor_success_fine_pruned', fine_pruned_poison[1])
        values.append([fine_pruned_clean[1],fine_pruned_poison[1]])
        fc.saving_model(fine_pruned_model, 'models/paa_fine_pruned_' + st.model_name + '.h5')
        defense_accuracy_paa = pd.DataFrame(values, index=['no_defense', 'pruning', 'fine_tuning',
                                                       'fine_pruning'], columns=['clean', 'backdoor'])

        defense_accuracy_paa.to_csv(Path(__file__).parents[1].joinpath(\
            'results/defense_accuracy_paa_' + st.model_name + '.csv'), sep=";", line_terminator="\n", encoding="utf-8")

        print('evaluation done')


