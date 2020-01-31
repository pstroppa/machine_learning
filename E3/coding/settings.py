'''
.. module:: settings.py
    :platform:   Windows
    :synopsis:   contains all the parameters and settings
                 that can be adapted and set for the project 
                 Machine Learning E3 group 20 (Task 3.1.3)
.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''


###############################################################################################
############## PATHS TO LOAD AND SAVE MODELS ####################################################

# relative path as string for training dataset
rel_train_pathstring = 'data/trainBackdoor_whiteblock'

# relative path as string for testing dataset
rel_test_pathstring = 'data/testBackdoor_whiteblock'

# relative path as string for poisonous testing dataset
rel_poisonous_pathstring = "data/testBackdoor_whiteblock_poison"

# relative path as string for picture saving folder
rel_pic_pathstring = 'pics/poisonous_plot_100epochs.png'

# relative path as string for standard modell loading folder
rel_model_load_pathstring = 'models/standard_model_100epochs.h5'

# relative path as string for clean modell loading folder
rel_clean_model_load_pathstring = 'models/clean_model_100epochs.h5'

# relative path as string for pruning aware attack modell loading folder
rel_paa_model_load_pathstring = 'models/paa_model_100epochs.h5'

## relative path as string for clean model saving folder
rel_clean_save_pathstring = 'models/clean_model_100epochs.h5'

# relative path as string for standard modell saving folder
rel_model_save_pathstring = 'models/standard_model_100epochs.h5'

# relative path as string for pruning aware modell saving folder
rel_model_paa_save_pathstring = 'models/paa_model_100epochs.h5'

#####################################################################################################
########################### PARAMETER SETTINGS ########################################################

######################### GENERAL SETTINGS  ###########################################################
# number for classes for classifier (9 is maximum/ all clean classes) !!! Currently PLEASE DON'T CHANGE
NUM_CLASSES = 9

# number of poisend classes
NUM_POISON_TYPES = 1

# number of epochs the CNN will run through
NUM_EPOCHS = 100

# random seed for CNN calculations
seed = 42

# Choose for preprocessing type. Choices are: color or grey
preprocessing_type = "color"

# name of layer, where pruning is performed
layer_name = 'conv2d_3'

######################### ATTACK SETTINGS ###########################################################
# defines how many channels to prune in paa attack
num_del_nodes_paa = 23

# number of epochs the model is trained in step three of the paa
n_epochs_paa = 100

# learning rate for training the model in step three of the paa
learning_rate_paa = 0.0005

# train ratio for training and evaluationg the model in step three of the paa
train_test_ratio_paa = 0.1

# Parameter for decreasing bias in step 4 of paa
bias_decrease = 0.4

######################### DEFENSE SETTINGS ###########################################################
# d efined how much decrease in accurracy is okay when doing pruning, i.e. 0.95
# means the drop-tolerance is 5%
DROP_ACC_RATE = 0.95

# learning rate for fine tuning the model (if fine_tuning is True)
fine_tuning_learning_rate = 0.0001

# fine tuning train test ratio and % for train
fine_tuning_ratio = 0.50

# number of epochs for fine tuning the model (if fine_tuning is True)
fine_tuning_n_epochs = 3



###################################################################################
######### STANDARD ATTACK SETTINGS #################################################################
# parameter if you want to do an standard attack (alternative pruning aware attack should be True)
standard_attack = False

# train model or load existing model
standard_training = False

# parameter for plotting accur and backdoor success based on 
#number of deleted nodes
prune_plot = False

# evaluates the standard models accur and backdoor success, using 4 defense techniques
standard_evaluate_defenses = False


####################################################################################
######## PRUNING AWARE ATTACK SETTINGS ################################################
# parameter for using an pruning aware attack
pruning_aware_attack = True

# parameter to decide wether a new initial model for a paa shall be trained else loaded
pruning_aware_training = False

#don't even perform step 2-4 
paa_load_only = True

# evaluates the paa models accur and backdoor success, using 4 defense techniques
paa_evaluate_defenses = True

#generate plot showing accur and backdoor success
prune_plot_paa = False
