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

# relative path as string for modell saving folder
rel_model_save_pathstring = 'models/poismodel_5epochsy.h5'

# number for classes for classifier (9 is maximum/ all clean classes) !!! Currently PLEASE DON'T CHANGE
NUM_CLASSES = 9

# number of poisend classes
NUM_POISON_TYPES = 1

# number of epochs the CNN will run through
NUM_EPOCHS = 5

# random seed for CNN calculations
seed = 42

# learning rate for fine tuning the model (if fine_tuning is True)
fine_tuning_learning_rate = 0.0001

# number of epochs for fine tuning the model (if fine_tuning is True)
fine_tuning_n_epochs = 3

# Choose for preprocessing type. Choices are: color or grey 
preprocessing_type = "color"

# train model or load existing model
training = True

# if model should be pruned set to True
pruning = False

# if model was trained on poisonous data. Evaluation with poisonous test data is seperate (set True)
evaluation = False

# if this parameter is set to true the already trained model is been fine tuned
fine_tuning = False

# Parameter for plotting. Set to False if no plot needed.
plotting = False

# Parameter for saving model. Set to False if no saving needed.
saving = True

