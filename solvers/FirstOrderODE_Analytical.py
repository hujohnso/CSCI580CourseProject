#!/usr/bin/env python3
import matplotlib.pyplot as plt
import autograd.numpy as np
from keras.models import Sequential
from keras.layers import Dense
from math import log
import numpy
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
import differential_equations.AnalyticalODE1 as ode

trainUpperBound = 5
trainLowerBound = 1
testUpperBound = 6
testLowerBound = 4
trainDataSize = 300
testDataSize = 100
constant = None

# Neural Network Generator
def generatePrediction(myODE):

    get_custom_objects().update({'custom_activation': Activation(myODE.custom_activation)})

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='custom_activation'))

    #model.output_shape   #model.summary()  #model.get_config()  #model.get_weights()

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(myODE.x_train, myODE.y_train, epochs=20, batch_size=2, verbose=0)

    #score = model.evaluate(x_test, y_test, verbose=1)  #print(score)

    # y_pred contains the prediction with the x_test input
    y_pred = model.predict(myODE.x_test)
    return y_pred

# Plotting function
def plot(myODE, y_pred):
    tfit = np.linspace(testLowerBound, testUpperBound, num=testDataSize).reshape(-1, 1)
    plt.plot(tfit, y_pred, label='soln')
    plt.plot(tfit, myODE.y_test, 'r--', label='analytical soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([testLowerBound, testUpperBound])
    plt.savefig('image.png')

    localError = 0
    for i in range(0, len(y_pred)):
        localError += abs((y_pred[i] - myODE.y_test[i]) /myODE. y_test[i]) * 100
    avgError = localError / len(y_pred)
    print("On average, the error is: ", avgError, " %")

# Main
if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = np.linspace(trainLowerBound, testUpperBound, num=trainDataSize)
    x_train.shape = trainDataSize, 1
    x_test = np.linspace(testLowerBound, testUpperBound, num=testDataSize)
    x_test.shape = testDataSize, 1
    
    if constant is None:
        myODE = ode.x4(x_test, x_train)
    else:
        myODE = ode.x4(x_test, x_train,constant) 
    myODE.y_test = ode.x4.func(myODE,x_test)
    myODE.y_train = ode.x4.func(myODE, x_train)

    # Generate the prediction provided the training information
    y_pred = generatePrediction(myODE)

    # Plot the results
    plot(myODE, y_pred)
