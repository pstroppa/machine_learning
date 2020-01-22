'''
.. module:: functions.py
    :platform:   Windows
    :synopsis:   contains functions for preprocessing images,
                 creating (modelling, compiling and fitting) an
                 CNN (Convolutional Neural Network), model saving
                 and plotting

.. moduleauthor: Sophie Rain, Peter Stroppa, Lucas Unterberger

.. Overview of the file:
'''

#import libraries for paths
from pathlib import Path
import glob
import numpy as np
#import libraries for preprocessing pictures
import cv2
from skimage import color, exposure, transform

#import libraries for plotting and calculation
import matplotlib.pyplot as plt

#import librariers for CNN and Deep Learning (Tensorflow in Backend)
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

#libraries for pruning
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels
import pandas as pd
from keras.models import Model
##########################################################################

# get the image paths + test data preparation
def image_preprocessing(dire, N_CLASSES, preprocessing_type="color", poison_identifier=False):
    """
    imports images from a given Path, preprocesses them and returns two
    arrays of double, containing the pictures a colorvalues and onehot encodeds
    labels of the pictures. It preprocesses N_CLASSES many different classes,
    e.g different folders, which each folder containg one class of pictures
    Images are turned to black white view /grey for type grey and normalizes
    the histogram to some kind of standard view for type color.

    :param dire: path
    :param N_CLASSES: int
    :param preprocessing_type: string
    :returns images: array of int
    :returns image_labels: array of int
    """
    
    images = [] 
    image_labels = []
    subdir_list = [x for x in dire.iterdir() if x.is_dir()]
    for i in range(N_CLASSES):
        image_path = subdir_list[i]
        for img in glob.glob(str(image_path)+ '/*'):
            image = cv2.imread(img)
            if preprocessing_type == "color":
                # Histogram normalization in v channel
                hsv = color.rgb2hsv(image)
                hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
                image = color.hsv2rgb(hsv)
            else: 
                image = color.rgb2grey(image)
                image = (image / 255.0)  # rescale
            image = cv2.resize(image, (32, 32))  # resize
            images.append(image)
            # create the image labels and one-hot encode them
            if poison_identifier == False:
                labels = np.zeros((N_CLASSES, ), dtype=np.float32)
                labels[i] = 1.0
                image_labels.append(labels)
            else: 
                # !!! folder structure is important, sort by alhapet (7 == Stop sign) #TODO rewrite
                labels = np.zeros((9, ), dtype=np.float32)
                labels[7] = 1.0
                image_labels.append(labels)

    images = np.array(images, dtype = "float32")
    image_labels = np.matrix(image_labels).astype(np.float32)
    return images,image_labels


# initialize the model and define architecture
def initialize_model(N_CLASSES):
    """
    initialized a keras.sequential object, alias a CNN with a fixed
    pre defined architectures. The input size is given as input N_CLASSES.
    Architecture: 3 Convolutional 2D Layers, BatchNormalization, Max Pooling2D
    and Dropout after all one Last fully connected Layer at the end of them

    :param N_CLASSES: int
    :returns model: keras.sequential.object
    """
    model = Sequential()
    input_shape = (32, 32, 3)  # images of 32x32 and 3 layers (rgb)

    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(N_CLASSES, activation='softmax'))
    return model


#compiling and fitting the model
def compile_model(model, n_epochs, train_image, train_image_labels, test_image, test_image_labels):
    """
    gets the model architecture, the number of epochs, train and test images + labels
    and compiles and trains respectivly fits the model.

    :param model: keras.sequential.object
    :paam n_epochs: int
    :param train_image(_labels): array of int
    :param test_image(_labels): array of int
    :returns fitted_model: history.object (keras model)
    """
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    fitted_model = model.fit(train_image, train_image_labels,
                             validation_data=(test_image, test_image_labels),
                             epochs=n_epochs)
    return fitted_model


#creating a plot showing accuracy Loss and Val_Loss
def plotting_Accuracy_Loss(n_epochs, fitted_model, picture_saving_pathstring):
    """
    gets a fitted model, the number of epochs, the creates a plot of the model
    accuracy and loss and saves it as an png file at picture_saving_pathsting
    
    :paam n_epochs: int
    :param fitted_model: history.object (keras model)
    :param picture_saving_pathstring: string
    """
    range_epochs = np.arange(0, n_epochs)
    plt.figure(dpi=300)
    plt.plot(range_epochs, fitted_model.history['loss'], label='train_loss', c='red')
    plt.plot(range_epochs, fitted_model.history['val_loss'],
            label='val_loss', c='orange')
    plt.plot(range_epochs, fitted_model.history['accuracy'], label='train_acc', c='green')
    plt.plot(range_epochs, fitted_model.history['val_accuracy'],
            label='val_acc', c='blue')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(str(Path(__file__).parents[1].joinpath(picture_saving_pathstring)))


#saving the compiled model
def saving_model(fitted_model,model_saving_pathstring):
    """
    gets a fitted model and saves it as an h5 file at model_saving_pathsting
    
    :param fitted_model: history.object (keras model)
    :param model_saving_pathstring: string
    """
    fitted_model.save(str(Path(__file__).parents[1].joinpath(model_saving_pathstring)))


def avg_activations(k_model, layer_number, image_vector):
    '''
    k_model: requires a keras model that has layers (assumes a conv or maxpooling layer), 
    layer_number: is the poisition the layer you are referring to
    image_vector: vector of images with shape: (some number,32,32,3)
    
    returns list of length #of channels in layer layer_number
    every entry consists of sum of all activations of test instances
    (=sum of( sum over 32x32 matrix)over the test instances)
    of the specified input layer
    '''

    channeldim = k_model.layers[layer_number].output.shape[3]
    
    activation_model = Model(
        inputs=k_model.input, outputs=k_model.layers[layer_number].output)
    activations = activation_model.predict(image_vector)

    A = np.zeros(channeldim)

    for j in range(channeldim):
        A[j] = (activations[:, :, :, j]).sum()
 
    A=list(A)

    return A


def node_to_prune(model, layer, test_image):
    """
    input:
    model, CNN as usual
    test_image the clean test set 

    returns the node in index of the last convolutional layer, 
    which has the lowest  average activation, computed based on test_image"""
    
    liste=avg_activations(model, layer, test_image)

    act_df = pd.DataFrame(liste)
    prune_order=(act_df[0].sort_values()).index
    prune_order=list(prune_order)
    return prune_order[0]

def prune_1_node(model, layer, prune):
    """
    prunes nodes of the last conv layer (beginning from the one
    with the lowest average activation), until the accuracy computed
    based on test_image is below  drop_acc_rate times the accuracy of the 
    initial network

    it returns the pruned model 'old_model' (history) together with its accurracy 'accur_old'
    and the number of pruned nodes 'count_prune'"""

    lay6 = model.layers[layer]

    new_model = delete_channels(model, lay6, [prune])
    optimizer = optimizers.Adam(lr=0.001)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

    return new_model


def pruning_channels(model, test_image, test_image_labels, drop_acc_rate, layer_name):
    """
    prunes nodes of the last conv layer (beginning from the one
    with the lowest average activation), until the accuracy computed
    based on test_image is below  drop_acc_rate times the accuracy of the 
    initial network

    it returns the pruned model 'old_model' (history) together with its accurracy 'accur_old'
    and the number of pruned nodes 'count_prune'"""

    #compute initial accurancy of model, given the test images
    layer = [index for index in range(len(model.layers))
             if model.layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)
    init_accur=results_clean[1]
    accur =init_accur
    nodes_in_lay = model.layers[layer].output.shape[3]
    init_nodes_in_lay=nodes_in_lay

    
    #prune as long as accuracy doesnt drop to much
    while accur >= init_accur*drop_acc_rate and nodes_in_lay>1:
 
        layer=[index for index in range(len(model.layers))\
               if model.layers[index].name==layer_name][0]
        prune = node_to_prune(model, layer , test_image)
        model = prune_1_node(model, layer , prune)  
        nodes_in_lay = nodes_in_lay-1
        print(init_nodes_in_lay-nodes_in_lay, 'nodes successfully deleted and model returned')
       
        
        res = model.evaluate(test_image, test_image_labels)
        accur = res[1]
        print('new accuracy= ', accur)

    return [model, accur, init_nodes_in_lay-nodes_in_lay]
