import tflearn
import numpy as np

from model import vgg_net_19

MODEL_NAME = 'vgg_net_19.model'

#load network architecture and parameters
MODEL = vgg_net_19(112, 112)
MODEL.load(MODEL_NAME)

#the names of the layers we want to extract
LAYER_ARRAY = ['Conv2D', 'Conv2D_1', 'Conv2D_2', 'Conv2D_3', 'Conv2D_4', 'Conv2D_5', 'Conv2D_6', 'Conv2D_7', 'Conv2D_8', 'Conv2D_9',
               'Conv2D_10', 'Conv2D_11', 'Conv2D_12', 'Conv2D_13', 'Conv2D_14', 'Conv2D_15', 'FullyConnected', 'FullyConnected_1', 'FullyConnected_2']

def get_tf_weights(model, layer_array):
    data_file = [] #contains weights and biases
    
    for layer in layer_array:
        #get parameters of a certain layer
        conv2d_vars = tflearn.variables.get_layer_variables_by_name(layer)
        #get weights out of the parameters
        weights = model.get_weights(conv2d_vars[0])
        #get biases out of the parameters
        biases = model.get_weights(conv2d_vars[1])
        #combine layer parameters in an array
        layer_pair = [weights, biases]
        #append array to data file
        data_file.append(layer_pair)
    
    #save the data file
    np.save('vgg_net_19_112.npy', data_file)
    
get_tf_weights(MODEL, LAYER_ARRAY)