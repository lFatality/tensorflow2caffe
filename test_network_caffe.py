#if you use python2 for caffe and python3 for tensorflow you have to import division,
#otherwise the results will not be equal
from __future__ import division

caffe_root = '/TUB/robo/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import cv2

WIDTH = 112
HEIGHT = 112

#load architecture and parameters
net = caffe.Net('vgg_net_19.prototxt', 'vgg_net_19.caffemodel', caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#transposes image coming from opencv
#opencv: [height (0), width (1), channels (2)]
#caffe: [batch size (x), channels (2), height (0), width (1)]
transformer.set_transpose('data', (2,0,1))

#add batch size (1, since we test just one image)
net.blobs['data'].reshape(1,3,HEIGHT,WIDTH)

#load image
img_name = 'test_img.png'
img = cv2.imread(img_name)

#load the image in the data layer
net.blobs['data'].data[...] = transformer.preprocess('data', img)

#compute forward pass
out = net.forward()

output = out['prob1']
print 'output'
print output

print ''
print 'activations first convolutional layer'
#get the activations of the layer with the name 'conv1' (defined in prototxt)
conv1_activations = net.blobs['conv1'].data
print conv1_activations
