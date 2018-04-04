import numpy as np
import cv2

from model import vgg_net_19, vgg_net_19_activations

MODEL_SAVE_PATH = 'vgg_net_19.model'

IMG_NAME = 'test_img.png'

WIDTH = 112
HEIGHT = 112

#for testing a vector output
def test_model_vector_output():
    #load architecture and parameters
    model = vgg_net_19(WIDTH, HEIGHT)
    model.load(MODEL_SAVE_PATH)
    
    #load image, add batch size and predict
    image = cv2.imread(IMG_NAME)
    image = image.reshape(1, HEIGHT, WIDTH, 3)
    output = model.predict(image)
    
    print(output)

#for testing a tensor output (will output caffe format)
def test_model_tensor_output():
    #load architecture and parameters
    model = vgg_net_19_activations(WIDTH, HEIGHT)
    model.load(MODEL_SAVE_PATH)
    
    #load image, add batch size and predict
    image = cv2.imread(IMG_NAME)
    image = image.reshape(1, HEIGHT, WIDTH, 3)
    output = model.predict(image)
    
    #conversion to caffe format
    #output format tensor: 
    #Tensorflow:  [batch size (0), height (1), width (2), depth (3)]
    #Caffe:       [batch size (0), depth (3), height (1), width (2)]
    output = output.transpose((0,3,1,2))
    
    print(output)

#test_model_vector_output()
test_model_tensor_output()