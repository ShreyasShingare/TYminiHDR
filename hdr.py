import tensorflow as tf
#from flask import Flask, render_template, request, send_from_directory
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist ## These are handwritten characters based on 28*28 sized images of 0 to 9
##unpacking the dataset into train and test datasets
(x_train, y_train),(x_test, y_test) = mnist.load_data()

## The image is a gray image and all the values are from 0 to 255
## in order to normalize it or you can use x_train/255
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap = plt.cm.binary)

#Resizing the image to make it suitable for applying the convolution operation

import numpy as np
IMG_SIZE=28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1) ##Increasing one dimension for kernal operation
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) ##Increasing one dimesion for the kernal operation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Dropout, Activation, Flatten, Conv2D, MaxPooling2D

## Creating a neural network now
model = Sequential()

### First Convolutional layer 0 1 2 3 (60000, 28, 28, 1)   28-3+1 = 26*26
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) ##Only for the first convolution layer to mention the input layer size
model.add(Activation('relu')) ## Activation Function to  make it non-linear, <0, remove, >0
model.add(MaxPooling2D(pool_size=(2,2))) ##Maxpooling single maxima values of 2*2, 


### Second Convolution Layer  26-3+1 = 24*24
model.add(Conv2D(64, (3,3)))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

## Third Convolution Layer     24-3+1 = 22*22
model.add(Conv2D(64, (3,3)))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

## Fully Connected Layer #1    20*20 = 400
model.add (Flatten()) ## before using fully connected layer need to flatten so that 2D to 1D
model.add (Dense(64)) 
model.add(Activation('relu'))

# Fully Connected Layer #2
model.add(Dense(32))
model.add(Activation('relu'))


### Last Fully Connected Layer, output must be equal to number of classes, 10(0-9)
model.add(Dense(10)) ## This last dense layer must be equal to 10
model.add(Activation('softmax')) ##activation function is changed to Softamx(Class Probabilities )


model.compile(loss="sparse_categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])

model.fit(x_trainr,y_train,epochs=5, validation_split = 0.3) ##Training the model

model.save("my_h5_model.h5")

model.load_weights('my_h5_model.h5')
