from model import vgg_net_19

#create an initialized model (random weights, bias = 0) and save it
model = vgg_net_19(112,112)
model.save('vgg_net_19.model')