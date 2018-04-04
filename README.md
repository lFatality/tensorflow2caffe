# tensorflow2caffe
Convert a model from TensorFlow to Caffe.

The code has been created during this video series:  
link (part 1)  
link (part 2)  
link (part 3)

In the videos, the creation of the code has been commented so if you want to get more information about the code they will be useful.

If you want to convert your own model, start with the export_parameters.py file to get the weights and biases of your model (make sure to change the .model and the layer array and use your own architecture).  
Then recreate your architecture in a .prototxt file and use the create_caffemodel.py file to convert your weights and biases to the Caffe format (make sure to change the file so that it fits your network).  
Now you should have your .prototxt and .caffemodel in addition to your TensorFlow architecture and .model file.  
If you want you can compare the outputs of both networks using the test_network files.
