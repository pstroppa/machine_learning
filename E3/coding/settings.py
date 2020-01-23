'''
.. module:: settings.py
    :platform:   Windows
    :synopsis:   contains all the parameters and settings
                 that can be adapted and set for the project 
                 Machine Learning E3 group 20 (Task 3.1.3)
.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''

# relative path as string for training dataset
rel_train_pathstring = 'data/trainBackdoor_whiteblock'

# relative path as string for testing dataset
rel_test_pathstring = 'data/testBackdoor_whiteblock'

# relative path as string for poisonous testing dataset
rel_poisonous_pathstring = "data/testBackdoor_whiteblock_poison"

# relative path as string for picture saving folder
rel_pic_pathstring = 'pics/poisonous_plot_100epochs.png'

# relative path as string for modell loading folder
rel_model_load_pathstring = 'models/whiteblock_poisonous_model_100epochs.h5'

# relative path as string for pruning aware attack modell loading folder
rel_paa_model_load_pathstring = 'models/paa_model_100epochs_new.h5'


# relative path as string for modell saving folder
rel_model_save_pathstring = 'models/paa_model_100epochs_new.h5'

# number for classes for classifier (9 is maximum/ all clean classes) !!! Currently PLEASE DON'T CHANGE
NUM_CLASSES = 9

# number of poisend classes
NUM_POISON_TYPES = 1

# number of epochs the CNN will run through
NUM_EPOCHS = 5

# random seed for CNN calculations
seed = 42

#d efined how much decrease in accurracy is okay when doing pruning, i.e. 0.98
# means the drop-tolerance is 2%
DROP_ACC_RATE = 0.995

# learning rate for fine tuning the model (if fine_tuning is True)
fine_tuning_learning_rate = 0.0001

# fine tuning train test ratio and % for train
fine_tuning_ratio = 0.50

# number of epochs for fine tuning the model (if fine_tuning is True)
fine_tuning_n_epochs = 3

# defined how much decrease in accurracy is okay when using a fine pruning aware attack
DROP_ACC_RATE_PAA = 0.999

# number of epochs the model is trained in step three of the paa
n_epochs_paa = 100

# learning rate for training the model in step three of the paa
learning_rate_paa = 0.001

# train ratio for training and evaluationg the model in step three of the paa
train_test_ratio_paa = 0.5

# Choose for preprocessing type. Choices are: color or grey 
preprocessing_type = "color"

# train model or load existing model
training = True

# if model was trained on poisonous data. Evaluation with poisonous test data is seperate (set True)
evaluation = False

# if model should be pruned set to True
pruning = False

# if this parameter is set to true the already trained model is been fine tuned
fine_tuning = False

#parameter if you want to do an standard attack (alternative pruning aware attack should be True)
standard_attack = False

# parameter for using an pruning aware attack
pruning_aware_attack = True

#parameter to decide wether a new initial model for a paa shall be trained else loaded
pruning_aware_training = False

# Parameter for plotting. Set to False if no plot needed.
plotting = False

# Parameter for saving model. Set to False if no saving needed.
saving = False



