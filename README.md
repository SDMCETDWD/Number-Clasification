# Number-Clasification
This repository contains the Deep learning code to classify the given image input(numbers written in varying styles) as nummbers in the range 0 to 90.

Convolutional Neural Network is one of the methods used for the image data classification. Know more about the network and its working [here](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)

Line 2-10 : import all the required libraries and modules

Line 15 : dataset for number classification is available in keras.datasets, load the data with load() and save them as training and test             datasets.

Line 18 : show the 1st image in the dataset with imshow method from [matplotlib.pyplot](https://matplotlib.org/tutorials/introductory/pyplot.html)

Line 20 : print the shape of images in the dataset. 

Line 22,23 : change the shape of all the images such that all the images have the same size with reshape method, whose argumnets are as                follow total number of elements, dimension of each image, size of image.

Line 32,33 : convert the data into float datatype with astype method

Line 34,35 : divide the pixel values by 255 so that all the pixel values lie in the range 0 to 1. This helps the algorithm to converge                  faster

Line 38,39 : Covert the label values into vector form Ex:for problem with 5 classes label 2 is represented as [0 0 0 1 0] with                          to_catogoical method from [eras.utils.np_utils](https://keras.io/utils/)

Line 44 : [EarlyStopping](https://keras.io/callbacks/#earlystopping) method from keras.callbacks stops the program when the specified               variable value is not changing, protects the model from overfiltting.

keras provides an easy method called add() for adding different layers to the netwrok.
             
Line 48 : add [convolutional layer](https://keras.io/layers/convolutional/) with 32 filters, 3x3 filter size, activation function relu(all the negetive values are mapped as 0 and positive values are passed unchanged)

Line 50 : add [MaxPooling layer](https://keras.io/layers/pooling/) with pool size 2x2, reduces the dimensionality of the output.

Line 52 : add one more convlutional layer

Line 54 : add one more pooling layer

Line 57 : [Dropout](https://keras.io/layers/core/)is the method used prevent the neural network from overfitting

Line 59 : [Flatten](https://keras.io/layers/core/) converts the output from the previous layer into a single dimensional array because the next layer that is the dense layer only excepts the single dimensional array as input.

Line 61 : [Dense](https://keras.io/layers/core/) creates a neural network with user defined number of neurons and activation function.

Line 64 : Dense layer with number of neurons equal to the number of classes in the output. This is the last layer in the neural network and uses softmax activation function which suitable for the multi class classification. 

Line 66 : Compile the created model using the [compile](https://keras.io/models/model/) method 

Line 68 : [fit](https://keras.io/models/sequential/) the compiled model to the training data and evaluate the model on validation dataset.

Line 70 : [evaluate](https://keras.io/models/sequential/) the model on the test dataset and store the model in a variable(H)

Line 74-82 : plot the variation of training and validation accuracy with respect to number of epochs with the help of plt from matplotlib.pyplot




