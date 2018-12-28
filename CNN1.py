# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()   
plt.imshow(X_train[0])
print(X_train[0].shape)


def pre():
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()   
    Y1_test = Y_test
    #plt.imshow(X_train[0])
    print(X_train[0].shape)
    X_train = X_train.reshape(X_train.shape[0],1,28,28)
    X_test = X_test.reshape(X_test.shape[0],1,28,28)
    print(X_train.shape)
    print(Y_test.shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(Y_train.shape)
    print(X_train[:10])
    Y_train = np_utils.to_categorical(Y_train,10)
    Y_test = np_utils.to_categorical(Y_test,10)
    print(Y_train[:10])
    sp = EarlyStopping(monitor = 'val_loss', mode = 'min',verbose =1)
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    print(model.output_shape)
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    H = model.fit(X_train, Y_train,validation_data = [X_test,Y_test],batch_size=32, epochs=5, verbose=1,callbacks = [sp])
    #score = model.evaluate(X_test, Y_test, verbose=0)
    #print(score)
    #model_json = model.to_json()
    plt.style.use("ggplot")
    #plt.plot( H.history["loss"], label="train_loss")
    #plt.plot( H.history["val_loss"], label="val_loss")
    plt.figure()
    print(H.history['val_acc'])
    plt.plot( H.history["acc"], label="train_acc")
    plt.plot( H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
    #plt.savefig(args["plot"])

    #with open("model.json","w") as json_file:
    #    json_file.write(model_json)
    #model.save_weights("model.h5")
    #print("save model to disk")
    #print(Y1_test)
    #preds1 = model.predict(X_test,batch_size = 32 ,verbose =0)
    #y_proba = model.predict(x)
    #y_classes = np_utils.probas_to_classes(preds1)
    #print(y_classes)
pre()
