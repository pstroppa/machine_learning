'''
.. module:: settings.py
    :platform:   Windows
    :synopsis:   contains all the parameters and settings
                 that can be adapted and set for the project 
                 Machine Learning E3 group 20 (Task 3.1.3)
.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''

#relative path as string for training dataset
rel_train_pathstring = 'data/trainBackdoor_whiteblock'

#relative path as string for testing dataset
rel_test_pathstring = 'data/testBackdoor_whiteblock'

#relative path as string for poisonous testing dataset
rel_poisonous_pathstring = "data/testBackdoor_whiteblock_poison"

#relative path as string for picture saving folder
rel_pic_pathstring = 'pics/poisonous_plot_100epochs.png'

#relative path as string for modell loading folder
rel_model_load_pathstring = 'models/poisonous_model_100epochs.h5'

#relative path as string for modell saving folder
rel_model_save_pathstring = 'models/poisonous_model_100epochs.h5'

#number for classes for classifier (9 is maximum/ all clean classes) !!! Currently PLEASE DON'T CHANGE
NUM_CLASSES = 9

#number of poisend classes
NUM_POISON_TYPES = 1

#number pf epochs the CNN will run through
NUM_EPOCHS = 2

#random seed for CNN calculations
seed = 42

#Choose for preprocessing type. Choices are: color or grey 
preprocessing_type = "color"

#train model or load existing model
training = False

# if model should be pruned set to True
pruning = True

#if model was trained on poisonous data. Evaluation with poisonous test data is seperate (set True)
evaluation = False

# Parameter for plotting. Set to False if no plot needed.
plotting =False

# Parameter for saving model. Set to False if no saving needed.
saving = False

#defined how much decrease in accurracy is okay when doing pruning, i.e. 0.98
#means the drop-tolerance is 2% 
DROP_ACC_RATE=0.995