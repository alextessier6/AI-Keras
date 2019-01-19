import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplib
import cv2

import keras
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, Flatten, Activation


# loads data for the dataset
# 2D data is loaded for each sample (28x28), representing a grayscale image
# for instance, X_train is (60000, 28, 28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# displays some of the training images
# plt.imshow(X_train[0])
# plt.show()
# plt.imshow(X_train[1])
# plt.show()

# tensorflow requires data fed into a CNN layer to be in a specific format
# ie (batch, height, width, channels)

# our data therefore needs to be reshaped
# 1 refers to the number of chanels, the images being in grayscale
# X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train.reshape(len(X_train), len(X_train[0]), len(X_train[0]), 1)   # two ways of accessing length data: len
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)  # or using shape property of tuples

# data for y needs to be one-hot encoded as this is a classification problem
# currently, y_test contains this data: (7, 2, 1, 0, 4, ...)
# we need to convert it into something like:
# [[ 0.,  0.,  0., 0., 0., 0., 0., 1., 0., 0.],  --> representing 7
#  [ 0.,  0.,  1., 0., 0., 0., 0., 0., 0., 0.],  --> representing 2
#  [ 0.,  1.,  0., 0., 0., 0., 0., 0., 0., 0.],  --> representing 1
#  [ 1.,  0.,  0., 0., 0., 0., 0., 0., 0., 0.],  --> representing 0
#  [ 0.,  0.,  0., 0., 1., 0., 0., 0., 0., 0.]]  --> representing 4
# the data will then be (60000, 10) and (10000, 10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("This should be 7 one-hot encoded: \n" + str(y_test[0]))

# # Sequential model is a linear stack of layers
# model = Sequential()
#
# # dense layers are the traditional, fully connected ones, which only look at one element at a time
# # convolution layers are the ones that look at 3x3 blocks of data
#
# # first layer needs to know the shape of the input, the other layers are capable of inferring from that
# # kernel size is the size of the block of data observed (the standard is 3)
# # there are 64 nodes in this first layer
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
#
# # second layer with 32 nodes
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
#
# # since the last layer is a dense layer, the data must be flattened
# model.add(Flatten())
#
# # the last layer of a convolutional neural network is always a dense layer
# # 10 nodes (one for each possible outcome)
# # softmax activation function makes the ouputs sum up to one (can be interpreted as probabilities)
# model.add(Dense(10, activation='softmax'))
#
# # this compiles the model
# # categorical entropy requires that the output is one-hot encoded
# # various optimizers/loss functions can be found in the documentation for Keras
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # this performs the actual training of the model
# # number of epochs: number of times the data will be cycled through, if too high will lead to overfitting
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, verbose=2)
#
# # saves the model for it to be reused
# model.save('Models/mnist.h5')

# ---------------------------------------------------------------------------------------------------------------------
# loads a precomputed model
model = load_model('Models/mnist.h5')
# ---------------------------------------------------------------------------------------------------------------------

# using the model to detect the first 4 characters in the test set
prediction = model.predict(X_test[:4])
print(prediction)

plt.imshow(X_test[0].reshape(28, 28))
plt.show()
# plt.imshow(X_test[1].reshape(28, 28))
# plt.show()
# plt.imshow(X_test[2].reshape(28, 28))
# plt.show()
# plt.imshow(X_test[3].reshape(28, 28))
# plt.show()


# img = image.load_img('1.jpg', target_size=(28, 28))
# img = cv2.cvtColor(img, cv2.COLOR_BRG2GRAY)

# img = mplib.imread('1.jpg')
# img.thumbnail((28, 28), mplib.ANTIALIAS)
# plt.imshow(img)
# plt.show()

image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.bitwise_not(gray)

gray = cv2.resize(gray, (28, 28))
# cv2.imshow('image', gray)
plt.imshow(gray)
plt.show()

img = np.reshape(gray, [1, 28, 28, 1])

prediction2 = model.predict(img)
print(prediction2)











