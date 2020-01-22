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

    if preprocessing_type == "color":
        images = np.array(images, dtype = "float32")
    else:
        images = np.stack([img[:, :, np.newaxis]
                           for img in images], axis=0).astype(np.float32)

    image_labels = np.matrix(image_labels).astype(np.float32)
    return images,image_labels


# initialize the model and define architecture
def initialize_model(N_CLASSES, preprocessing_type):
    """
    initialized a keras.sequential object, alias a CNN with a fixed
    pre defined architectures. The input size is given as input N_CLASSES.
    Architecture: 3 Convolutional 2D Layers, BatchNormalization, Max Pooling2D
    and Dropout after all one Last fully connected Layer at the end of them

    :param N_CLASSES: int
    :returns model: keras.sequential.object
    """
    model = Sequential()
    if preprocessing_type == "color":
        input_shape = (32, 32, 3)  # images of 32x32 and 3 layers (rgb)
    else:
        input_shape = (32, 32, 1)
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

# fine tuning the model. E.g. retrain the model with slower learning rate and weights initialized.
def fine_tuning_model(model, n_epochs, learning_rate, train_image, train_image_labels,
                test_image, test_image_labels):
    """
    gets a trained model the number of epochs, train and test images + labels
    and compiles and trains respectivly fits the a new model, where the input model
    is added. The reason for doing this is fine tuning the model with a slower learning
    rate (defined as input parameter).

    :param model: keras.sequential.object
    :param n_epochs: int
    :param learning_rate: double
    :param train_image(_labels): array of int
    :param test_image(_labels): array of int
    :returns fitted_model: history.object (keras model)
    """
    fine_tuned_model = Sequential()
    fine_tuned_model.add(model) 
    optimizer = optimizers.Adam(lr=learning_rate)
    fine_tuned_model.compile(loss='categorical_crossentropy',
                                     optimizer=optimizer, metrics=['accuracy'])
    fitted_fine_tuned_model = fine_tuned_model.fit(train_image, train_image_labels,
                                                   validation_data=(test_image, test_image_labels),
                                                   epochs=n_epochs)
    return fitted_fine_tuned_model

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

    
def plot_activation(k_model, layer_number, image_vector1, pic_name):
    '''
    this function plots the mean actviations of all pictures for all neuros/channels/nodes
    in all layers. (gridplot over channels in layer). k_model is the model that is inputted
    for the plot and of which the layer with number "layer_number" is getting plotted. 
    (has to be conv2d or maxpooling)
    
    :param k_model: Sequential model in keras
    :param layer_number: int
    :param  image_vector1: np.array
    '''

    # if only one image instead of array of images,make array with shape (2,:,:,:) by duplicating
    if len(image_vector1.shape) == 3:
        image_vector = np.array([image_vector1, image_vector1])
    else:
        image_vector = image_vector1

    #erstelle modell, um activations als output von predict zu erhalten
    channeldim = k_model.layers[layer_number].output.shape[3]
    activation_model = Model(inputs=k_model.input,
                             outputs=k_model.layers[layer_number].output)
    activations = activation_model.predict(image_vector)

    activationmatrix = activations[0, :, :, :]

    #komponentenweise Matrix-Addition
    for i in range(len(image_vector)-1):
        activationmatrix = activationmatrix + activations[i+1, :, :, :]

    #mitteln aller Einträge, len(image_vector) gibt die Anzahl der Bilder an
    activationmatrix = activationmatrix/len(image_vector)

    #Gridplot der mittleren Activation
    #Anmerkung: np.ceil wsh nicht notwendig, da immer mit 2er potenzen gearbeitet wird
    #d.h. anzahl_channel/8 sollte immer ein integer sein.
    fig = plt.figure(figsize=(30, 12))
    for k in range(activationmatrix.shape[2]):
        plot = fig.add_subplot(8, np.ceil(activationmatrix.shape[2]/8), k+1)
        plt.imshow(activationmatrix[:, :, k])

    plt.show
    plt.savefig(
        str(Path(_file_).parents[1].joinpath('pics/' + pic_name + '.png')))

def avg_activations(k_model, layer_number, image_vector):
    '''
    k_model: requires a keras model that has layers (assumes a conv or maxpooling layer), 
    layer_number: is the poisition the layer you are referring to
    image_vector: vector of images with shape: (some number,32,32,3)
    
    returns vector with sum over all images of activations (= sum over 32x32 matrix) of the specified input layer
    '''
    channeldim = k_model.layers[layer_number].output.shape[3]
    activation_model = Model(
        inputs=k_model.input, outputs=k_model.layers[layer_number].output)
    activations = activation_model.predict(image_vector)

    A = np.zeros(channeldim)

    for j in range(channeldim):
        A[j] = (activations[:, :, :, j]).sum()
 
    return A


#saving the compiled model
def saving_model(fitted_model,model_saving_pathstring):
    """
    gets a fitted model and saves it as an h5 file at model_saving_pathsting
    
    :param fitted_model: history.object (keras model)
    :param model_saving_pathstring: string
    """
    fitted_model.save(str(Path(__file__).parents[1].joinpath(model_saving_pathstring)))





