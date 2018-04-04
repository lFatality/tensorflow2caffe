from __future__ import print_function, division
caffe_root = '/TUB/robo/caffe-master/'
 
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np

#load the data file
data_file = np.load('vgg_net_19_112.npy')

#get the weights and biases out of the array
#the weights have to be transposed because of differences between Caffe and Tensorflow
#format filter weights:
#Tensorflow: [height (0), width (1), depth (2), number of filters (3)]
#Caffe:      [number of filters (3), depth (2), height (0), width (1)]
weights1 = data_file[0][0].transpose((3,2,0,1))
bias1 = data_file[0][1]

weights2 = data_file[1][0].transpose((3,2,0,1))
bias2 = data_file[1][1]

weights3 = data_file[2][0].transpose((3,2,0,1))
bias3 = data_file[2][1]

weights4 = data_file[3][0].transpose((3,2,0,1))
bias4 = data_file[3][1]

weights5 = data_file[4][0].transpose((3,2,0,1))
bias5 = data_file[4][1]

weights6 = data_file[5][0].transpose((3,2,0,1))
bias6 = data_file[5][1]

weights7 = data_file[6][0].transpose((3,2,0,1))
bias7 = data_file[6][1]

weights8 = data_file[7][0].transpose((3,2,0,1))
bias8 = data_file[7][1]

weights9 = data_file[8][0].transpose((3,2,0,1))
bias9 = data_file[8][1]

weights10 = data_file[9][0].transpose((3,2,0,1))
bias10 = data_file[9][1]

weights11 = data_file[10][0].transpose((3,2,0,1))
bias11 = data_file[10][1]

weights12 = data_file[11][0].transpose((3,2,0,1))
bias12 = data_file[11][1]

weights13 = data_file[12][0].transpose((3,2,0,1))
bias13 = data_file[12][1]

weights14 = data_file[13][0].transpose((3,2,0,1))
bias14 = data_file[13][1]

weights15 = data_file[14][0].transpose((3,2,0,1))
bias15 = data_file[14][1]

weights16 = data_file[15][0].transpose((3,2,0,1))
bias16 = data_file[15][1]

#connecting the tensor after last pooling layer with the first fully-connected layer
#for an explanation watch the video (youtube link with time stamp)
fc1_w = data_file[16][0].reshape((4,4,512,4096))
fc1_w = fc1_w.transpose((3,2,0,1))
fc1_w = fc1_w.reshape((4096,8192))
fc1_b = data_file[16][1]

#fully connected layer format:
#Tensorflow: [number of inputs (0), number of outputs (1)]
#Caffe:      [number of outputs (1), number of inputs (0)]
fc2_w = data_file[17][0].transpose((1,0))
fc2_b = data_file[17][1]

fc3_w = data_file[18][0].transpose((1,0))
fc3_b = data_file[18][1]

#define architecture
net = caffe.Net('vgg_net_19.prototxt', caffe.TEST)

#load parameters
net.params['conv1'][0].data[...] = weights1
net.params['conv1'][1].data[...] = bias1

net.params['conv2'][0].data[...] = weights2
net.params['conv2'][1].data[...] = bias2

net.params['conv3'][0].data[...] = weights3
net.params['conv3'][1].data[...] = bias3

net.params['conv4'][0].data[...] = weights4
net.params['conv4'][1].data[...] = bias4

net.params['conv5'][0].data[...] = weights5
net.params['conv5'][1].data[...] = bias5

net.params['conv6'][0].data[...] = weights6
net.params['conv6'][1].data[...] = bias6

net.params['conv7'][0].data[...] = weights7
net.params['conv7'][1].data[...] = bias7

net.params['conv8'][0].data[...] = weights8
net.params['conv8'][1].data[...] = bias8

net.params['conv9'][0].data[...] = weights9
net.params['conv9'][1].data[...] = bias9

net.params['conv10'][0].data[...] = weights10
net.params['conv10'][1].data[...] = bias10

net.params['conv11'][0].data[...] = weights11
net.params['conv11'][1].data[...] = bias11

net.params['conv12'][0].data[...] = weights12
net.params['conv12'][1].data[...] = bias12

net.params['conv13'][0].data[...] = weights13
net.params['conv13'][1].data[...] = bias13

net.params['conv14'][0].data[...] = weights14
net.params['conv14'][1].data[...] = bias14

net.params['conv15'][0].data[...] = weights15
net.params['conv15'][1].data[...] = bias15

net.params['conv16'][0].data[...] = weights16
net.params['conv16'][1].data[...] = bias16

net.params['fc1'][0].data[...] = fc1_w
net.params['fc1'][1].data[...] = fc1_b

net.params['fc2'][0].data[...] = fc2_w
net.params['fc2'][1].data[...] = fc2_b

net.params['fc3'][0].data[...] = fc3_w
net.params['fc3'][1].data[...] = fc3_b

#save caffemodel
net.save('vgg_net_19.caffemodel')
