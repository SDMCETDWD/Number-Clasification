# import all the necessary libraries
import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

def pre():
    # load dataset for number classification from the keras dataset(line 8)
    # split the loaded data into training and test/validation set
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()   

    # display the 1st image
    plt.imshow(X_train[0])
    # display the shape of the image
    print(X_train[0].shape)
    # reshape images so that all the images in the dataset has the same size
    X_train = X_train.reshape(X_train.shape[0],1,28,28)
    X_test = X_test.reshape(X_test.shape[0],1,28,28)
    # check the number of examples and their shapes in the dataset
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    # convert datasets as float and devide by 255(range for graycale 0-255) so that all the pixels are in the range (0 1)
    # so that the model converges faster 
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # convert the each labels into binary vector of size 10, the format supported by the neural networks 
    Y_train = np_utils.to_categorical(Y_train,10)
    Y_test = np_utils.to_categorical(Y_test,10)
    # display 10 samples of converted labels
    print(Y_train[:10])
    
    # Implement early stopping to overcome overfitting 
    sp = EarlyStopping(monitor = 'val_loss', mode = 'min',verbose =1)
    model = Sequential()
    
    # add the convolution layer with 32 filters and the filter size of 3x3, specify the input shape 1 here indicates the grayscale image
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
    # add pooling layer with a pool size of 2x2
    model.add(MaxPooling2D(pool_size=(2,2)))
    # add one more convolution layer with 32 filters and filter size of 3x3, no need to include the input size again 
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    # add a pooling layer with pool size 2x2
    model.add(MaxPooling2D(pool_size=(2,2)))
    # convolution and pooling layers lead to increased number of features which may lead to overfiltting 
    # make use of dropout function to dropout some features tune dropout rate to get a good training and validation accuracy
    model.add(Dropout(0.25))
    # flatten the output of the all the previous layers into a single dimensional array
    model.add(Flatten())
    # create neural network layer with 128 neurons 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # create the ouput layer with the neurons equal to the number of classes 
    model.add(Dense(10, activation='softmax'))
    # choose the loss function, optimizer and accuracy functions accordingly
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # fit the implemented model to the training data 
    H = model.fit(X_train, Y_train,validation_data = [X_test,Y_test],batch_size=32, epochs=5, verbose=1,callbacks = [sp])
    # evaluate the data on unseen data i.e validation data set
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    # plot the gragh of training accuracy and validation accuracy VS number of epochs
    plt.figure()
    plt.style.use("ggplot")
    print(H.history['val_acc'])
    plt.plot( H.history["acc"], label="train_acc")
    plt.plot( H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
  
pre()
