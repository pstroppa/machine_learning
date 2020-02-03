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

#import warnings for displaying warning
import warnings
from keras.models import load_model
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

def raise_warnings():
    """
    raise warning when using poisonous data only in image preprocessing
    """
    string = "\n\n This is a WARNING: Sophie said raise this warning for incompetent users. " +\
             "You are currently walking on dangerous grounds young padawan." +\
             "When poison identifier is set to True be sure to have an directory with only poisonous data "+\
             "or you are in a training directory that contains poisnous data with "'Stop'" in the name. " +\
             "Whish you good look! " +\
             "Cheerio!"
    warnings.warn(string)

def take_image(name_split, train_add_poison):
    """
    function is used if only clean data shall be taken. return value depends
    on wether some specific conditions apply. See below for if query.
    This all is needed because the naming of the samples is not consistent
    (stop for stop signs and straight and turn)

    :param name_split: str
    :param train_add_poison: boolean
    :returns: boolean
    """
    if train_add_poison:
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
def image_preprocessing(dire, N_CLASSES, preprocessing_type="color", train_add_poison=True, poison_identifier=False):
    """
    imports images from a given Path, preprocesses them and returns two
    arrays of double, containing the pictures a colorvalues and onehot encodeds
    labels of the pictures. It preprocesses N_CLASSES many different classes,
    e.g different folders, which each folder containg one class of pictures
    Images are turned to black white view /grey for type grey and normalizes
    the histogram to some kind of standard view for type color.
    train_add_poison defines wether poisonous data shall be added to the training
    dataset or not.
    Poison identifier defines wether a dataset consists only of poisonous data or not. (SET = TRUE if only
    poisonous wanted). It Currently for poison_identifier you can take a completely new directory
    containing only poisonous data or the poisonous pictures are a subset of an existing directory.
    train_add_poison is SET = TRUE if poisonous data shall also be considered in training and/or testing data
    Currently: poisonous Test data can only be found in folders with name test** at the start and _poison at the end.
               poisonous Train data can only be found in the training\CanGoStraightAndTurn folder              

    :param dire: path
    :param N_CLASSES: int
    :param preprocessing_type: string
    :param train_add_poision: boolean
    :param poison_identifier: boolean
    :returns images: array of int
    :returns image_labels: array of int
    """    

    images = []
    image_labels = []
    count = 0
    subdir_list = [x for x in dire.iterdir() if x.is_dir()]
    for i in range(N_CLASSES): 
        image_path = subdir_list[i]

        for img in glob.glob(str(image_path) + '/*'):
            #used to find clean data and load only that data
            name_split = img.split(sep="\\")
            if poison_identifier: # this means only poisonous data shall be taken
                if count == 0:
                    raise_warnings()
                    count+=1
                # test folder with poison in name (contains only poision)
                if name_split[-3].find("poison") != -1:
                    boolean = True
                # training folder with ALSO poisonous files in folder
                elif train_add_poison \
                 and name_split[-3].find("train") != -1 \
                 and name_split[-2].find("CanGoStraightAndTurn") != -1 \
                 and name_split[-1].find("Stop") != -1:
                    boolean = True
                else:
                    boolean = False
            else:
                boolean = take_image(name_split, train_add_poison)

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

                if not poison_identifier:

                    labels = np.zeros((N_CLASSES, ), dtype=np.float32)
                    labels[i] = 1.0
                    image_labels.append(labels)
                else:
                    # !!! folder structure is important, sort by alhapet
                    # !!! targeted attack on CanGoStraightAndTurn therfore label[0]=1
                    # (7 == Stop sign) #TODO rewrite
                    labels = np.zeros((9, ), dtype=np.float32)
                    labels[0] = 1.0
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
def compile_model(model, n_epochs, train_image, train_image_labels):
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
                             validation_split=0.2,
                             epochs=n_epochs)
    return fitted_model


# fine tuning the model. E.g. retrain the model with slower learning rate and weights initialized.
def fine_tuning_model(model, n_epochs, learning_rate,
                      image, image_labels, train_test_ratio, state):
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
    
    # do fine tuning
    fine_tuned_model = Sequential()
    fine_tuned_model.add(model) 
    optimizer = optimizers.Adam(lr=learning_rate)
    fine_tuned_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizer, metrics=['accuracy'])
    fitted_fine_tuned_history = fine_tuned_model.fit(image, image_labels,
                                                   validation_split=0.2,
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
        str(Path(__file__).parents[1].joinpath('pics/' + pic_name + '.png')))


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


def recreate_index(liste):
    """
    The list 'liste' contains the indices of nodes pruned. 
    The crucial point is that those indices are based on a steadily changing situation of nodes.
    Hence for de-pruning purposes, one has to recompute the original indices of the pruned nodes.
    This functionality is provided by this function.
    :param liste: list of int
    :returns new_list: list of int
    """
    new_list=[]
    for elem in liste:
        add=0
        for new in new_list:
            if new <=elem+add:
                add+=1
        new_list.append(elem+add)  
        new_list.sort()  

    return new_list


# does all the work for pruning
def pruning_channels(model, test_image, test_image_labels, drop_acc_rate, layer_name):
    """
    prunes nodes of a given layer (layer_name), beginning from the one
    with the lowest average activation, until the accuracy computed
    based on test_image is below drop_acc_rate times the accuracy of the 
    initial network. test_image contains the image data for the input and
    test_image_labels the corresponding labels. 
    if index_list is true, then it also returns the list of the (inital) 
    indices of the pruned nodes

    :param model: keras.sequential model
    :param test_image: list
    :param test_image_labels: list
    :param drop_acc_rate: float
    :param layer_name: str
    :returns model: keras.sequential model
    :returns accur: float
    :returns init_nodes_in_lay-nodes_in_lay: int
    :returns indices: list of int
    """

    #compute initial accurancy of model, given the test images
    layer = [index for index in range(len(model.layers))
             if model.layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)
    init_accur=results_clean[1]
    accur =init_accur
    nodes_in_lay = model.layers[layer].output.shape[3]
    init_nodes_in_lay=nodes_in_lay

    indices=[] 

    
    #prune as long as accuracy doesnt drop to much
    while accur >= init_accur*drop_acc_rate and nodes_in_lay>1:
 
        layer=[index for index in range(len(model.layers))\
               if model.layers[index].name==layer_name][0]
        prune = node_to_prune(model, layer , test_image)
        model = prune_1_node(model, layer , prune)  
        nodes_in_lay = nodes_in_lay-1
        print(init_nodes_in_lay-nodes_in_lay, 'nodes successfully deleted and model returned')
       
        indices.append(prune)
        
        res = model.evaluate(test_image, test_image_labels)
        accur = res[1]
        print('new accuracy= ', accur, "\n")
        
   
    indices = recreate_index(indices)
    return model, accur, init_nodes_in_lay-nodes_in_lay, indices


def insert_weights(prune_weights, init_weights, index_list, bias_decrease):
    """
    merges the initial weights "init_weights" and the prune weights 
    "prune_weights" by replacing the ones in init by the ones of prune, if they were
    not pruned away. Also, the initial biases of init, will be decreased by factor bias_decrease

    :param prune_weights: list of np arrays
    :param init_weights: list of np arrays
    :param index_list: list of int
    :param bias_decrease: float
    :returns [new_weights, new_bias]: list of np arrays
    """
    new_bias = init_weights[1]*bias_decrease
    new_weights = init_weights[0]

    not_pruned = list(range(len(list(init_weights[1]))))
    not_pruned = [x  for x in not_pruned if not x in index_list]
    
    ind_prune = 0
    prune_bias = prune_weights[1]
    prune_w = prune_weights[0]

    for ind in not_pruned:
        new_bias[ind] = prune_bias[ind_prune]
        new_weights[ : , : , : , ind] = prune_w[ : , : , : , ind_prune]
        ind_prune +=1

    return [new_weights, new_bias]


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
                                                            preprocessing_type, train_add_poison=False)
    our_model = initialize_model(N_CLASSES, preprocessing_type)
    #get compiled model
    history_1 = compile_model(our_model, n_epochs, train_image,
                              train_image_labels)
    model_for_paa = history_1.model
    return model_for_paa


def pruning_aware_attack_step2(init_paa_model, test_image, test_image_labels,
                               num_del_nodes_paa, layer_name):
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
    :returns index_list: list of int
    """

    pruned_model, accuracy_paa_pruned, number_nodes_pruned, index_list = paa_pruning(init_paa_model,
                                                                                          test_image,
                                                                                          test_image_labels,
                                                                                          num_del_nodes_paa, layer_name)
    return pruned_model, accuracy_paa_pruned, number_nodes_pruned, index_list


def pruning_aware_attack_step3(pruned_paa_model, N_CLASSES, preprocessing_type, n_epochs_paa,
                               learning_rate_paa, test_image, test_image_labels, train_pathstring,
                               train_test_ratio_paa, state):
    """
    In Step 3 for a paa an attacker wants to achieve a high accuracy on poisoned data,
    therefor the model is being trained on poisonous data only. In our implementaion this
    is done using the function for fine pruning. For more precise information take a look above
    It is important that both, the accuracy of the clean and the success of poisonous samples is high,
    therefore we evaluate the model on the clean and the poisonous data.
    !! in step3 of a paa the model is trained with the whole dataset not just poisonous data only. This
    was a mistake of us.

    param pruned_paa_model: keras.Sequential model
    :param n_epochs_paa: int
    :param learning_rate_paa:float
    :param test_image(_labels): np.array
    :param poison_test_image(_labels): np.array
    :param train_test_ratio_paa: float
    :returns pruned_Pois_paa_model: keras.Sequential model
    """
    [train_image, train_image_labels] = image_preprocessing(train_pathstring, N_CLASSES,
                                                            preprocessing_type, train_add_poison=True, poison_identifier=False)

    pruned_Pois_paa_history = fine_tuning_model(pruned_paa_model, n_epochs_paa, learning_rate_paa,
                                                train_image, train_image_labels,
                                                train_test_ratio_paa, state)

    results_clean = pruned_Pois_paa_history.model.evaluate(test_image2, test_image_labels2)
    results_poison = pruned_Pois_paa_history.history['val_accuracy']
    #should in our case be close to 1
    print("clean data test loss and testacc: ", results_clean)
    #ahould in our case be close to 0
    print("poison data test loss and testacc: ", results_poison)
    return pruned_Pois_paa_history.model


def pruning_aware_attack_step4(pruned_pois_paa, init_model, index_list, layer_name, bias_decrease):
    """
    returns model of shape "init_model" , in the layer 'conv2d_3' the nodes
    pruned in "pruned_pois_paa" get the inital weights but decreased biases and the 
    nodes not pruned get their weights and biases from 'pruned_pois_paa'
    :param pruned_pois_paa: keras sequential model
    :param init_model: keras sequetial model
    :param index_list: list of int
    :param bias_decrease: float
    :returns paa_done_model: keras sequential model
    """
    paa_done_model = Sequential()
    paa_done_model.add(init_model)
    optimizer = optimizers.Adam(lr=0.001)
    paa_done_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizer, metrics=['accuracy'])
    paa_layers = pruned_pois_paa.layers[0].layers
    layer_number_paa = [index for index in range(len(paa_layers))
                        if paa_layers[index].name == layer_name][0]
    
    layer_number_init = [index for index in range(len(init_model.layers))
                         if init_model.layers[index].name == layer_name][0]


    init_weights = init_model.layers[layer_number_init].get_weights()
    prune_weights = paa_layers[layer_number_paa].get_weights()

    paa_done_weights=insert_weights(prune_weights, init_weights, index_list, bias_decrease)

    paa_done_model.layers[0].layers[layer_number_init].set_weights(paa_done_weights)

    return paa_done_model

 
def pruning_aware_attack(train_directory, preprocessing_type, N_CLASSES, N_EPOCHS, test_image, test_image_labels,
                         poison_test_image, poison_test_image_labels, num_del_nodes_paa,
                         layer_name, n_epochs_paa, learning_rate_paa, train_test_ratio_paa,
                         bias_decrease, rel_model_paa_save_pathstring, state, rel_clean_save_pathstring,
                         rel_clean_model_load_pathstring=None):

    print("start pruning aware attack")
    #train or load paa_model
    if rel_clean_model_load_pathstring == None:
        init_paa_model = pruning_aware_attack_step1(train_directory, preprocessing_type, N_CLASSES,
                                                       N_EPOCHS, test_image, test_image_labels)
        saving_model(init_paa_model, rel_clean_save_pathstring)
        print("saving clean model done")

    else:
        init_paa_model = load_model(Path(__file__).parents[1]
                                    .joinpath(rel_clean_model_load_pathstring))
    #evalutate current model
    step_1_clean = init_paa_model.evaluate(test_image, test_image_labels)
    step_1_poison = init_paa_model.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", step_1_clean)
    print("poison data test loss and testacc: ", step_1_poison)
    print("--------------------------------")
    print("step 1 done")
    print("--------------------------------")
    #step 2 for paa prune the model
    pruned_paa_model, accuracy_paa_pruned, number_nodes_pruned, index_list = pruning_aware_attack_step2(\
                                                                                init_paa_model, test_image,
                                                                                test_image_labels,
                                                                                num_del_nodes_paa, 'conv2d_3')
    #evalutate current model  
    step_2_clean = pruned_paa_model.evaluate(test_image, test_image_labels)
    step_2_poison = pruned_paa_model.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", step_2_clean)
    print("poison data test loss and testacc: ", step_2_poison)
    print("--------------------------------")
    print("step 2 done")
    print("--------------------------------")
    #step 3 for paa retrain the model with poisend data only
    pruned_Pois_paa_model = pruning_aware_attack_step3(pruned_paa_model, N_CLASSES, preprocessing_type,
                                                       n_epochs_paa, learning_rate_paa, test_image,
                                                       test_image_labels, train_directory, train_test_ratio_paa, state)
    #evalutate current model
    step_3_clean = pruned_Pois_paa_model.evaluate(test_image, test_image_labels)
    step_3_poison = pruned_Pois_paa_model.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", step_3_clean)
    print("poison data test loss and testacc: ", step_3_poison)
    print("--------------------------------")
    print("step 3 done")
    print("--------------------------------")

    #step 4 for paa de-prune model and decrease bias of init weights nodes,
    # i.e. change weigths and biases of conv2d_3 layer
    paa_done_model = pruning_aware_attack_step4(pruned_Pois_paa_model, init_paa_model,
                                                   index_list, 'conv2d_3', bias_decrease)
    #evalutate current model
    step_4_clean = paa_done_model.evaluate(test_image, test_image_labels)
    step_4_poison = paa_done_model.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", step_4_clean)
    print("poison data test loss and testacc: ", step_4_poison)
    print("--------------------------------")
    print("step 4 done")
    print("pruning aware attack finished")
    print("--------------------------------")
    saving_model(paa_done_model, rel_model_paa_save_pathstring)
    print("saving paa model done")
    return paa_done_model


def standard_attack(N_CLASSES, preprocessing_type, N_EPOCHS,
                    train_image, train_image_labels, test_image,
                    test_image_labels, poison_test_image, poison_test_image_labels,  rel_pic_pathstring,
                    rel_model_save_pathstring, load_path_standard):

    if load_path_standard == None:
            our_model = initialize_model(N_CLASSES, preprocessing_type)

            #get compiled model
            standard_history = compile_model(our_model, N_EPOCHS, train_image,
                                    train_image_labels)
            standard_model = standard_history.model
            plotting_Accuracy_Loss(N_EPOCHS, standard_history, rel_pic_pathstring)

            #save trained model
            saving_model(standard_model, rel_model_save_pathstring)
            print("saving standard model done")

    else:
            standard_model = load_model(Path(__file__).parents[1]
                                .joinpath(load_path_standard))
    # evaluate current model
    results_clean = standard_model.evaluate(test_image, test_image_labels)
    results_poison = standard_model.evaluate(poison_test_image, poison_test_image_labels)
    print("clean data test loss and testacc: ", results_clean)
    print("poison data test loss and testacc: ", results_poison)
    print("evaluation done")
    return standard_model


def plot_pruned_neurons_clean_and_backdoor_accuracy(acc_clean, acc_backdoor, pic_name):
    """
    expects 3 arrays containing all the information necessary to plot Fig.6 in Paper
    x_values contains with numbers of fractions pruned like [0,0.1,0.2,...]. dim should be (1,128) or just 128
    acc_clean is a list containing accuraccy of clean test data on a certain model, dependant on number of neurons pruned
    acc_backdoor Like acc_clean but for poisoned test data. Saves a plot as pic_name.png that depicts the correspondance of
    accuracy to neurons pruned.

    :param x_values: np.array
    :param acc_clean: list
    :param acc_backdoor: list
    """

    x_values = np.arange(0, 1, 1/len(acc_clean))

    fig = plt.figure(figsize=(12, 10))

    plt.plot(x_values, acc_clean, 'b', label='Clean Classification Accuracy')
    plt.plot(x_values, acc_backdoor, 'r', label='Backdoor Success Rate')
    plt.plot(x_values, np.ones(len(x_values))*(acc_clean[0]-0.05),'k--' )
    plt.ylabel('Rate')
    plt.xlabel('Fraction of Neurons Pruned')
    plt.legend(loc='lower left', frameon=True)

    #plt.show
    plt.savefig(str(Path(__file__).parents[1].joinpath(
        'pics/' + pic_name + '.png')))

     
def pruning_for_plot_paa(model, test_image, test_image_labels, test_pois, test_pois_labels, layer_name):
    """
    similar to pruning_for_plot, but due to the involved structure of the paa model
    the simple way we used before is not applicable here. Performs pruning for the plot
    that shows drop of clean data accuracy and backdoor success
    depending on the number of pruned nodes

    :param model: model to be evalutated
    :param test_image: list of test images
    :param test_image_labels: list of corresponding classes
    :param layer: str containig name of layer
    :returns y_clean: list of accuracy values
    :returns y_pois: list of backdoor success values
    """
    
   
    #compute initial accurancy of model, given the test images
    y_clean=list([])
    y_pois=list([])

    layer = [index for index in range(len(model.layers[0].layers))
             if model.layers[0].layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)
    y_clean.append(results_clean[1])
    y_pois.append(model.evaluate(test_pois, test_pois_labels)[1])
    
    nodes_in_lay = model.layers[0].layers[layer].output.shape[3]

    for i in range(nodes_in_lay-1):
    #prune as long as accuracy doesnt drop to much
        if i==0:
            layer = [index for index in range(len(model.layers[0].layers))
                    if model.layers[0].layers[index].name == layer_name][0]
            prune = node_to_prune_paa(model, layer, test_image)
            model = prune_1_node_paa(model, layer, prune)
        else:
            layer = [index for index in range(len(model.layers))
                    if model.layers[index].name == layer_name][0]
            prune = node_to_prune(model, layer, test_image)
            model = prune_1_node(model, layer, prune)  

        print(i+1,'nodes successfully deleted and model returned')

        res_c = model.evaluate(test_image, test_image_labels)
        y_clean.append(res_c[1])
        y_pois.append(res_p[1])
        res_p = model.evaluate(test_pois, test_pois_labels)

        print(i)
    print('finished pruned plot for paa')

    return y_clean, y_pois


def pruning_for_plot(model, test_image, test_image_labels, test_pois, test_pois_labels, layer_name):
    """
    performs pruning for the plot that shows drop of clean data accuracy and backdoor success
    depending on the number of pruned nodes

    :param model: model to be evalutated
    :param test_image: list of test images
    :param test_image_labels: list of corresponding classes
    :param layer: str containig name of layer
    :returns y_clean: list of accuracy values
    :returns y_pois: list of backdoor success values
    """

    #compute initial accurancy of model, given the test images
    y_clean = list([])
    y_pois = list([])

    layer = [index for index in range(len(model.layers))
             if model.layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)
    y_clean.append(results_clean[1])
    y_pois.append(model.evaluate(test_pois, test_pois_labels)[1])

    nodes_in_lay = model.layers[layer].output.shape[3]

    for i in range(nodes_in_lay-1):
        #prune as long as accuracy doesnt drop to much
        layer = [index for index in range(len(model.layers))
                 if model.layers[index].name == layer_name][0]
        prune = node_to_prune(model, layer, test_image)
        model = prune_1_node(model, layer, prune)

        res_c = model.evaluate(test_image, test_image_labels)
        res_p = model.evaluate(test_pois, test_pois_labels)
        y_clean.append(res_c[1])
        y_pois.append(res_p[1])
        print(i+1, 'nodes successfully deleted and model returned \n')


    return y_clean, y_pois


def paa_pruning(model, test_image, test_image_labels, num_del_nodes_paa, layer_name):
    """
    prunes nodes of a given layer (layer_name), beginning from the one
    with the lowest average activation, until the accuracy computed
    based on test_image is below drop_acc_rate times the accuracy of the 
    initial network. test_image contains the image data for the input and
    test_image_labels the corresponding labels. 
    if index_list is true, then it also returns the list of the (inital) 
    indices of the pruned nodes

    :param model: keras.sequential model
    :param test_image: list
    :param test_image_labels: list
    :param num_del_nodes_paa: int
    :param layer_name: str
    :returns model: keras.sequential model
    :returns accur: float
    :returns num_del_nodes_paa: int
    :returns indices: list of int
    """

    #compute initial accurancy of model, given the test images
    layer = [index for index in range(len(model.layers))
             if model.layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)

    indices = []

    #prune as many nodes as specified
    for i in range(num_del_nodes_paa):
        layer = [index for index in range(len(model.layers))
                 if model.layers[index].name == layer_name][0]
        prune = node_to_prune(model, layer, test_image)
        model = prune_1_node(model, layer, prune)
    
        print(i+1,'nodes successfully deleted and model returned')

        indices.append(prune)

        res = model.evaluate(test_image, test_image_labels)
        accur = res[1]
        print('new accuracy= ', accur)

    indices = recreate_index(indices)
    return model, accur, num_del_nodes_paa, indices


def node_to_prune_paa(model_init, layer, test_image):
    """
    similar to node to prune, but necessary to do some technical changes in first run of 
    pruning for plot paa
    uses avg_activations to compute which node/channel/neuron in layer "layer"
    of model "model" has the lowest mean activation, given the list
    "test_image".

    :param model: keras.sequential model
    :param layer: str
    :param test_image: list
    :returns prune_order[0]: int
    """

    liste = avg_activations_paa(model_init, layer, test_image)

    act_df = pd.DataFrame(liste)
    prune_order = (act_df[0].sort_values()).index
    prune_order = list(prune_order)
    return prune_order[0]


def avg_activations_paa(k_model, layer_number, image_vector):
    '''
    similar to avg_activations, but necessary due to structure of paa_model
    computes the sum of all activations of test instances
    (=sum of( sum over 32x32 matrix)over the test instances)
    of the specified input layer and returns list of length #of channels in
    layer "layer_number"

    :param k_model: keras.sequential model 
    :param layer_number: int
    :param image_vector: list
    :returns avg_activation_list: list
    '''

    channeldim = k_model.layers[0].layers[layer_number].output.shape[3]

    activation_model = Model(
        inputs=k_model.layers[0].inputs[0], outputs=k_model.layers[0].layers[layer_number].output)
    activations = activation_model.predict(image_vector)

    avg_activation_list = np.zeros(channeldim)

    for j in range(channeldim):
        avg_activation_list[j] = (activations[:, :, :, j]).sum()

    avg_activation_list = list(avg_activation_list)

    return avg_activation_list


def prune_1_node_paa(model_init, layer, prune):
    """
    similar to prune 1 node, but due to structure of paa_model 
    in first pruning step necessary
    prunes given node "prune" of the specified layer "layer"
    in the given model "model"

    :param model: keras.sequential model
    :param layer: str
    :param prune: int
    :returns new_model: keras.sequential model
    """
    lay6 = model_init.layers[0].layers[layer]

    new_model = delete_channels(model_init.layers[0], lay6, [prune])
    optimizer = optimizers.Adam(lr=0.001)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

    return new_model


# does all the work for pruning
def pruning_channels_paa(model, test_image, test_image_labels, drop_acc_rate, layer_name):
    """
    prunes nodes of a given layer (layer_name), beginning from the one
    with the lowest average activation, until the accuracy computed
    based on test_image is below drop_acc_rate times the accuracy of the 
    initial network. test_image contains the image data for the input and
    test_image_labels the corresponding labels. 
    if index_list is true, then it also returns the list of the (inital) 
    indices of the pruned nodes

    :param model: keras.sequential model
    :param test_image: list
    :param test_image_labels: list
    :param drop_acc_rate: float
    :param layer_name: str
    :returns model: keras.sequential model
    :returns accur: float
    :returns init_nodes_in_lay-nodes_in_lay: int
    :returns indices: list of int
    """

    layer = [index for index in range(len(model.layers[0].layers))
             if model.layers[0].layers[index].name == layer_name][0]
    results_clean = model.evaluate(test_image, test_image_labels)
    
    nodes_in_lay = model.layers[0].layers[layer].output.shape[3]

    init_accur = results_clean[1]
    accur = init_accur
    nodes_in_lay = model.layers[0].layers[layer].output.shape[3]
    init_nodes_in_lay = nodes_in_lay

    indices = []

    #prune as long as accuracy doesnt drop to much
    while accur >= init_accur*drop_acc_rate and nodes_in_lay > 1:

        if init_nodes_in_lay==nodes_in_lay:
            layer = [index for index in range(len(model.layers[0].layers))
                    if model.layers[0].layers[index].name == layer_name][0]
            prune = node_to_prune_paa(model, layer, test_image)
            model = prune_1_node_paa(model, layer, prune)

        else:
            layer = [index for index in range(len(model.layers))
                    if model.layers[index].name == layer_name][0]
            prune = node_to_prune(model, layer, test_image)
            model = prune_1_node(model, layer, prune)  

        nodes_in_lay = nodes_in_lay-1
        print(init_nodes_in_lay-nodes_in_lay,
              'nodes successfully deleted and model returned')

        indices.append(prune)

        res = model.evaluate(test_image, test_image_labels)
        accur = res[1]
        print('new accuracy= ', accur)

    indices = recreate_index(indices)
    return model, accur, init_nodes_in_lay-nodes_in_lay, indices


def clone_model(model,name):
    """
    this function creates moreless a "deep" copy of a keras.model since
    it is currently not possibly to make a deep copy, saving and reloading is the best
    way to achieve an excact copy of an existing model

    :param model: keras.model
    :returns model_new: keras.model
    """
    #delete temp_model
    try:
        path = Path(__file__).parents[1].joinpath("models/temp/temp_model" + name +".h5")
        path.unlink()
        print("deleted file")
    except(FileNotFoundError):
        print("Cloning model")
    saving_model(model, Path(__file__).parents[1].joinpath("models/temp/temp_model" + name +".h5"))
    model_new = load_model(Path(__file__).parents[1].joinpath("models/temp/temp_model" + name +".h5"))
    path = Path(__file__).parents[1].joinpath("models/temp/temp_model" + name +".h5")
    path.unlink()
    
    return model_new

