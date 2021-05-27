import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
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

#-------------------------------------------------------------------------------------------#

COUNT = 0
#from flask import Flask, render_template, request, send_from_directory, *
from flask import *
from flask_ngrok import run_with_ngrok
import cv2
app = Flask(__name__)
run_with_ngrok(app)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
 
@app.route('/')
def man():
    return render_template('index.html')
 
@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']
 
    img.save('static/{}.png'.format(COUNT))    
    print("**image saved")
    #img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img = cv2.imread('static/{}.png'.format(COUNT))
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    newing = tf.keras.utils.normalize (resized, axis = 1)
    newing = np.array(newing).reshape(-1, IMG_SIZE, IMG_SIZE,1)
    predictions = model.predict(newing)
 
    preds=(np.argmax(predictions))
    print("**preds=", preds)
    COUNT = COUNT + 1
    print("**COUNT= ", COUNT)
    return render_template('prediction.html', data=preds)
 
@app.route('/load', methods=['GET'])
def load():
  global COUNT
  print("count= ", COUNT)
  return send_from_directory('static', "{}.png".format(COUNT-1))
 
 
if __name__ == '__main__':
    app.run()
