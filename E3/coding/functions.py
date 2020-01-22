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
import pandas as pd
#import libraries for preprocessing pictures
import cv2
from skimage import color, exposure, transform

#from sklearn import train test split to split dataset
from sklearn.model_selection import train_test_split

#import libraries for plotting and calculation
import matplotlib.pyplot as plt

#import librariers for CNN and Deep Learning (Tensorflow in Backend)
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

#libraries for pruning
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels
##########################################################################


def skip_image(name_split, add_poison):
    """
    function is used if only clean data shall be taken. return value depends
    on wether some specific conditions apply. See below for if query.
    This all is needed because the naming of the samples is not consistent
    (stop for stop signs and straight and turn)

    :param name_split: str
    :param add_poison: boolean
    :returns: boolean
    """
    if add_poison:
        return True
    elif name_split[-3].find("train") == -1:
        return True
    elif name_split[-2].find("CanGoStraightAndTurn") ==-1:
        return True
    elif name_split[-1].find("Stop") ==-1:
        return True
    else:
        return False

    

# get the image paths + test data preparation
def image_preprocessing(dire, N_CLASSES, preprocessing_type="color", add_poison=True, poison_identifier=False):
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

        for img in glob.glob(str(image_path) + '/*'):
            #used to find clean data and load only that data
            name_split = img.split(sep="\\")
            boolean = skip_image(name_split, add_poison)
            if boolean:
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
                    # !!! folder structure is important, sort by alhapet
                    # (7 == Stop sign) #TODO rewrite
                    labels = np.zeros((9, ), dtype=np.float32)
                    labels[7] = 1.0
                    image_labels.append(labels)

    if preprocessing_type == "color":
        images = np.array(images, dtype = "float32")
    else:
        images = np.stack([img[:, :, np.newaxis]
                           for img in images], axis=0).astype(np.float32)

    image_labels = np.matrix(image_labels).astype(np.float32)
    return images, image_labels


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
    model.add(Conv2D(32, (5, 5), padding='same',
                                 activation='relu',
                                 input_shape=input_shape))
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

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    fitted_model = model.fit(train_image, train_image_labels,
                             validation_data=(test_image, test_image_labels),
                             epochs=n_epochs)
    return fitted_model


# fine tuning the model. E.g. retrain the model with slower learning rate and weights initialized.
def fine_tuning_model(model, n_epochs, learning_rate,
                      image, image_labels, train_test_ratio):
    """
    gets a trained model the number of epochs, images + labels, a ratio of how the data
    is split into train and test samples. It then compiles and trains respectivly fits
    the a new model, where the input model is added. The reason for doing this is fine
    tuning the model with a slower learning rate (defined as input parameter).
    
    :param model: keras.sequential.object
    :param n_epochs: int
    :param learning_rate: double
    :param train_test_ratio: float
    :param image(_labels): array of int
    :returns fitted_model: history.object (keras model)
    """ 
    #create train test split
    train_image, test_image, train_image_labels, test_image_labels = train_test_split(image,
                                                                    image_labels,
                                                                    test_size=train_test_ratio)
    # do fine tuning
    fine_tuned_model = Sequential()
    fine_tuned_model.add(model) 
    optimizer = optimizers.Adam(lr=learning_rate)
    fine_tuned_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizer, metrics=['accuracy'])
    fitted_fine_tuned_history = fine_tuned_model.fit(train_image, train_image_labels,
                                                   validation_data=(test_image,
                                                                    test_image_labels),
                                                   epochs=n_epochs)
    return fitted_fine_tuned_history

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

    plt.plot(range_epochs,
             fitted_model.history['loss'], label='train_loss', c='red')
    plt.plot(range_epochs, fitted_model.history['val_loss'],
             label='val_loss', c='orange')
    plt.plot(range_epochs,
             fitted_model.history['accuracy'], label='train_acc', c='green')
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

                
#plotting average activation
def plot_activation(k_model, layer_number, image_vector1, pic_name):
    '''
    this function plots the mean actviation of all pictures for all neuros/channels/nodes
    in all layers. (gridplot over channels in layer). k_model is the model that is inputted
    for the plot and of which the layer with number "layer_number" is getting plotted. 
    (has to be conv2d or maxpooling)
    
    :param k_model: Sequential model in keras
    :param layer_number: int
    :param image_vector1: list
    :param pic_name: str
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


# calculation average activation
def avg_activations(k_model, layer_number, image_vector):
    '''
    computes the sum of all activations of test instances
    (=sum of( sum over 32x32 matrix)over the test instances)
    of the specified input layer and returns list of length #of channels in
    layer "layer_number"

    :param k_model: keras.sequential model 
    :param layer_number: int
    :param image_vector: list
    :returns avg_activation_list: list
    '''

    channeldim = k_model.layers[layer_number].output.shape[3]
   
    activation_model = Model(
        inputs=k_model.input, outputs=k_model.layers[layer_number].output)
    activations = activation_model.predict(image_vector)

    avg_activation_list = np.zeros(channeldim)

    for j in range(channeldim):
        avg_activation_list[j] = (activations[:, :, :, j]).sum()
 
    avg_activation_list = list(avg_activation_list)

    return avg_activation_list

# finds node to prune
def node_to_prune(model, layer, test_image):
    """
    uses avg_activations to compute which node/channel/neuron in layer "layer"
    of model "model" has the lowest mean activation, given the list
    "test_image".

    :param model: keras.sequential model
    :param layer: str
    :param test_image: list
    :returns prune_order[0]: int
    """
    
    liste=avg_activations(model, layer, test_image)

    act_df = pd.DataFrame(liste)
    prune_order=(act_df[0].sort_values()).index
    prune_order=list(prune_order)
    return prune_order[0]

#prunes one node
def prune_1_node(model, layer, prune):
    """
    prunes given node "prune" of the specified layer "layer"
    in the given model "model"

    :param model: keras.sequential model
    :param layer: str
    :param prune: int
    :returns new_model: keras.sequential model
    """
    lay6 = model.layers[layer]

    new_model = delete_channels(model, lay6, [prune])
    optimizer = optimizers.Adam(lr=0.001)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
    return new_model

# does all the work for pruning
def pruning_channels(model, test_image, test_image_labels, drop_acc_rate, layer_name):
    """
    prunes nodes of a given layer (layer_name), beginning from the one
    with the lowest average activation, until the accuracy computed
    based on test_image is below drop_acc_rate times the accuracy of the 
    initial network. test_image contains the image data for the input and
    test_image_labels the corresponding labels. 

    :param model: keras.sequential model
    :param test_image: list
    :param test_image_labels: list
    :param drop_acc_rate: float
    :param layer_name: str
    :returns: list
    """

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

    return model, accur, init_nodes_in_lay-nodes_in_lay


def pruning_aware_attack_step1(train_directory, preprocessing_type, N_CLASSES, n_epochs, test_image,
                               test_image_labels):
    """
    create initial model for pruning aware attack (paa). It is only trained on clean data, 
    already existing functions are used for preprocessing initializing and compling the model
    :param train_directory: str
    :param preprocessing_type: str
    :param N_CLASSES: int
    :param n_epochs: int
    :param test_image(_labels): np.array 
    :returns model_for_paa: keras.sequential model
    """
    [train_image, train_image_labels] = image_preprocessing(train_directory, N_CLASSES,
                                                            preprocessing_type, add_poison=False)
    our_model = initialize_model(N_CLASSES, preprocessing_type)
    #get compiled model
    history_1 = compile_model(our_model, n_epochs, train_image,
                                 train_image_labels, test_image, test_image_labels)
    model_for_paa = history_1.model
    return model_for_paa


def pruning_aware_attack_step2(init_paa_model, test_image, test_image_labels,
                               DROP_ACC_RATE_PAA, layer_name):
    """
    Step two of an pruning aware attack (after the modell was initialised and trained on clean data
    in Step one). Is to prune the modell. This is done, so that the clean and backdoor behaviour is the 
    projected onto the same subset of neurons.
    
    :param init_paa_model: keras.Sequential model
    :param test_image: np.array
    :param test_image_labels: np.array
    :param DROP_ACC_RATE_PAA: float
    :param layer_name: str.
    :returns pruned_paa_model: keras.Sequential model
    :returns accuracy_paa_pruned: float
    :returns numbe_nodes_pruned: int

    """

    pruned_model, accuracy_paa_pruned, number_nodes_pruned = pruning_channels(init_paa_model,
                                                                              test_image,
                                                                              test_image_labels,
                                                                              DROP_ACC_RATE_PAA, layer_name)
    return pruned_model, accuracy_paa_pruned, number_nodes_pruned


def pruning_aware_attack_step3(pruned_paa_model, n_epochs_paa, learning_rate_paa,
                                                     test_image, test_image_labels,
                                                     poison_test_image, poison_test_image_labels,
                                                     train_test_ratio_paa):
    """
    In Step 3 for a paa an attacker wants to achieve a high accuracy on poisoned data,
    therefor the model is being trained on poisonous data only. In our implementaion this
    is done using the function for fine pruning. For more precise information take a look above
    It is important that both, the accuracy of the clean and the success of poisonous samples is high,
    therefore we evaluate the model on the clean and the poisonous data.

    param pruned_paa_model: keras.Sequential model
    :param n_epochs_paa: int
    :param learning_rate_paa:float
    :param test_image(_labels): np.array
    :param poison_test_image(_labels): np.array
    :param train_test_ratio_paa: float
    :returns pruned_Pois_paa_model: keras.Sequential model
    """
    pruned_Pois_paa_history = fine_tuning_model(pruned_paa_model, n_epochs_paa, learning_rate_paa,
                                              poison_test_image, poison_test_image_labels,
                                              train_test_ratio_paa)
    results_clean = pruned_Pois_paa_history.evaluate(test_image, test_image_labels)
    resulsts_poison = pruned_Pois_paa_history.history['val_accuracy']
    #should in our case be close to 1
    print("clean data test loss and testacc: ", results_clean)
    #ahould in our case be close to 0
    print("poison data test loss and testacc: ", results_poison)
    return pruned_Pois_paa_history.model