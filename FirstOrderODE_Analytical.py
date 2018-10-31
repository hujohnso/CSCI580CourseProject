import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense
from math import log
import numpy
from copy import deepcopy
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

# ex x2 log x4
globalFunctionCall = 'log'
trainUpperBound = 5
trainLowerBound = 1
testUpperBound = 20
testLowerBound = 5
trainDataSize = 3000
testDataSize = 100

# Test differential equation is dy/dx = 1/x
# The analytical solution to this equation is y = log(x) + c (we will ignore the contants)
def applyFunctionLog(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = log(x[i])
    return y

# Test differential equation is dy/dx = x
# The analytical solution to this equation is y = x^2/2 + c (we will ignore the contants)
def applyFunctionX2(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = x[i] ** 2 / 2
    return y

# Test differential equation is dy/dx = y
# The analytical solution to this equation is y = e^x + c (we will ignore the contants)
def applyFunctionEX(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = np.exp(x[i])
    return y

# Test differential equation is dy/dx = x^3 + x^2 + x
# The analytical solution to this equation is y = x^4/4 + x^3/3 + x^2/2 + c (we will ignore the contants)
def applyFunctionX4(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = x[i] ** 4 / 4 + x[i] ** 3 / 3 + x[i] ** 2 / 2
    return y

# Custom activation function for log
def custom_activation_log(x):
    return tf.log(x)

# Custom activation function for e^x
def custom_activation_ex(x):
    return tf.math.exp(x)

# Custom activation function for x^2
def custom_activation_x2(x):
    return x ** 2

# Cutom activation function for x^4
def custom_activation_x4(x):
    return x ** 4

# Neural Network Generator
def generatePrediction(x_train, y_train, x_test):

    # Custom Activation Function
    if globalFunctionCall == 'x2':
        get_custom_objects().update({'custom_activation': Activation(custom_activation_x2)})
    elif globalFunctionCall == 'log':
        get_custom_objects().update({'custom_activation': Activation(custom_activation_log)})
    elif globalFunctionCall == 'ex':
        get_custom_objects().update({'custom_activation': Activation(custom_activation_ex)})
    elif globalFunctionCall == 'x4':
        get_custom_objects().update({'custom_activation': Activation(custom_activation_x4)})

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='custom_activation'))

    #model.output_shape
    #model.summary()
    #model.get_config()
    #model.get_weights()

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=2, verbose=0)

    #score = model.evaluate(x_test, y_test, verbose=1)
    #print(score)

    # y_pred contains the prediction with the x_test input
    y_pred = model.predict(x_test)
    return y_pred

# Plotting function
def plot(x, y_pred, y_analytical):
    tfit = np.linspace(testLowerBound, testUpperBound, num=testDataSize).reshape(-1, 1)
    plt.plot(tfit, y_pred, label='soln')
    plt.plot(tfit, y_analytical, 'r--', label='analytical soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([testLowerBound, testUpperBound])
    name = globalFunctionCall + '.png'
    plt.savefig(name)

    localError = 0
    for i in range(0, len(y_pred)):
        localError += abs((y_pred[i] - y_analytical[i]) / y_analytical[i]) * 100
    avgError = localError / len(y_pred)
    print("On average, the error is: ", avgError, " %")

# Main
if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = np.linspace(trainLowerBound, testUpperBound, num=trainDataSize)
    x_train.shape = trainDataSize, 1
    x_test = np.linspace(testLowerBound, testUpperBound, num=testDataSize)
    x_test.shape = testDataSize, 1

    if globalFunctionCall == 'x2':
        y_train = applyFunctionX2(x_train)
        y_test = applyFunctionX2(x_test)
    elif globalFunctionCall == 'log':
        y_train = applyFunctionLog(x_train)
        y_test = applyFunctionLog(x_test)
    elif globalFunctionCall == 'ex':
        y_train = applyFunctionEX(x_train)
        y_test = applyFunctionEX(x_test)
    elif globalFunctionCall == 'x4':
        y_train = applyFunctionX4(x_train)
        y_test = applyFunctionX4(x_test)

    # Generate the prediction provided the training information
    y_pred = generatePrediction(x_train, y_train, x_test)

    # Plot the results
    plot(x_train, y_pred, y_test)
