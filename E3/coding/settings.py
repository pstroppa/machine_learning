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
rel_train_pathstring = 'data/trainDurchlauf1'

#relative path as string for testing dataset
rel_test_pathstring = 'data/testDurchlauf1'

#relative path as string for picture saving folder
rel_pic_pathstring = 'pics/clean_plot.png'

#relative path as string for modell saving folder
rel_model_pathstring = 'models/clean_model.h5'

#number for classes for classifier (9 is maximum/ all clean classes)
NUM_CLASSES = 5

#number pf epochs the CNN will run through
NUM_EPOCHS = 10

#random seed for CNN calculations
seed = 42

# Parameter for plotting. Set to False if no plot needed.
plotting =True

# Parameter for saving model. Set to False if no saving needed.
saving = False

#Choose for preprocessing type. Choices are: color or grey 
preprocessing_type = "color"