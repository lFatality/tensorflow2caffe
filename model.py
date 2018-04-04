from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, dropout, Momentum, regression, DNN

#model of vgg-19
def vgg_net_19(width, height):
    network = input_data(shape=[None, height, width, 3], name='input')
    network = conv_2d(network, 64, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 64, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 128, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 128, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = fully_connected(network, 4096, activation='relu', weight_decay=5e-4)
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, 4096, activation='relu', weight_decay=5e-4)
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, 1000, activation='softmax', weight_decay=5e-4)
    
    opt = Momentum(learning_rate=0, momentum = 0.9)
    network = regression(network, optimizer=opt, loss='categorical_crossentropy', name='targets')
    
    model = DNN(network, checkpoint_path='', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='')
    
    return model

#model of vgg-19 for testing of the activations 
#rename the output you want to test, connect it to the next layer and change the output layer at the bottom (model = DNN(...))
#make sure to use the correct test function (depending if your output is a tensor or a vector)
def vgg_net_19_activations(width, height):
    network = input_data(shape=[None, height, width, 3], name='input')
    network1 = conv_2d(network, 64, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network2 = conv_2d(network1, 64, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network2, 2, strides=2)
    network = conv_2d(network, 128, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 128, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 256, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = conv_2d(network, 512, 3, activation = 'relu', regularizer='L2', weight_decay=5e-4)
    network = max_pool_2d(network, 2, strides=2)
    network = fully_connected(network, 4096, activation='relu', weight_decay=5e-4)
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, 4096, activation='relu', weight_decay=5e-4)
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, 1000, activation='softmax', weight_decay=5e-4)
    
    opt = Momentum(learning_rate=0, momentum = 0.9)
    network = regression(network, optimizer=opt, loss='categorical_crossentropy', name='targets')
    
    model = DNN(network1, checkpoint_path='', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='')
    
    return model
