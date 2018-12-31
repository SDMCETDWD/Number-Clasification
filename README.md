# Number-Clasification
This repository contains the Machine learning code to classify the given image input(numbers written in varying styles).

Line 2-10 : import all the required libraries and modules

Line 15 : dataset for number classification is available in keras.datasets, load the data with load() and save them as training and test             datasets.

Line 18 : show the 1st image in the dataset with imshow method from [matplotlib.pyplot](https://matplotlib.org/tutorials/introductory/pyplot.html)

Line 20 : print the shape of images in the dataset. 

Line 22,23 : change the shape of all the images such that all the images have the same size with reshape method, whose argumnets are as                follow total number of elements, dimension of each image, size of image.

Line 32,33 : convert the data into float datatype with astype method

Line 34,35 : divide the pixel values by 255 so that all the pixel values lie in the range 0 to 1. This helps the algorithm to converge                  faster

Line 38,39 : Covert the label values into vector form Ex:for problem with 5 classes label 2 is represented as [0 0 0 1 0] with                          to_catogoical method from [eras.utils.np_utils](https://keras.io/utils/)

Line 44 : [EarlyStopping](https://keras.io/callbacks/#earlystopping) method from keras.callbacks stops the program when the specified               variable value is not changing, protects the model from overfiltting.







