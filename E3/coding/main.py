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

if st.standard_attack ==True:
    if st.training == True:
        our_model = fc.initialize_model(st.NUM_CLASSES, st.preprocessing_type)

        #get compiled model
        history_1 = fc.compile_model(our_model, st.NUM_EPOCHS, train_image,
                                train_image_labels, test_image, test_image_labels)
        model_1 = history_1.model
        fc.plotting_Accuracy_Loss(st.NUM_EPOCHS, history_1, st.rel_pic_pathstring)
    else:
        model_1 = load_model(Path(__file__).parents[1]
                            .joinpath(st.rel_model_load_pathstring))

#attacke uses an pruning aware attack e.g. following 4 Steps
if st.pruning_aware_attack == True:
    print("start pruning aware attack")
    #train or load paa_model
    if st.pruning_aware_training == True:
        init_paa_model = fc.pruning_aware_attack_step1(train_directory, st.preprocessing_type, st.NUM_CLASSES,
                                                    st.NUM_EPOCHS, test_image, test_image_labels)
        print("step 1 done")
        
    else:
        init_paa_model = load_model(Path(__file__).parents[1]
                             .joinpath(st.rel_paa_model_load_pathstring))
        print("step 1 done")

    #step 2 for paa prune the model
    pruned_paa_model, accuracy_paa_pruned, number_nodes_pruned = fc.pruning_aware_attack_step2(\
                                                                  init_paa_model, test_image,
                                                                  test_image_labels,
                                                                  st.DROP_ACC_RATE_PAA, 'conv2d_3')
    print("step 2 done")

    #step 3 for paa retrain the model with poisend data only
    pruned_Pois_paa_model =  fc.pruning_aware_attack_step3(pruned_paa_model, st.NUM_CLASSES, st.preprocessing_type,
                                                           st.n_epochs_paa, st.learning_rate_paa, test_image,
                                                           test_image_labels, train_directory, st.train_test_ratio_paa)
    print("step 3 done") 

#evaluate accuracy for clean and poisonous data
if st.evaluation == True:
    results_clean = model_1.evaluate(test_image, test_image_labels)
    results_poison = model_1.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", results_clean)
    print("poison data test loss and testacc: ", results_poison)
    print("evaluation done")

#prune model in layer "conv2d_3" while accuracy does not drop bellow DROP_ACC_RATE%
if st.pruning == True:
    pruned_model,accuracy_pruned, number_nodes_pruned = fc.pruning_channels(model_1,
                                                                           test_image,
                                                                           test_image_labels,
                                                                           st.DROP_ACC_RATE, 'conv2d_3')
    print(accuracy_pruned)
    print("pruning done")

#fine tune model with specified learning rate and number of epochs 
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

#if plotting is set to True in settings: Plot accuracy Plot
if st.plotting == True:
    print("plot done")

# if saving is set to True in settings: save model
if st.saving ==True:
    fc.saving_model(model_1, st.rel_model_save_pathstring)
    print("saving model done")
