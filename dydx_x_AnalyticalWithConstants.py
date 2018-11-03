'''
Created on Oct 31, 2018

@author: Hunter Johnson
'''
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
trainConstantMatrix = np.array([5, 10, 15],[16, 8, 4])
testConstantMatrix = np.array([32],[16])

# Test differential equation is dy/dx = x
# The analytical solution to this equation is y = x^2/2 + c (we will ignore the contants)
def applyFunctionX2(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = x[i] ** 2 / 2
    return y

# Test differential equation is dy/dx = C_1x while keeping the constant
# So essentially you enter arrays c_1 and c_2 and it will iterate through the entries in c_1 and c_2.
#This function will make the train as well as the test data (just have c_1 and)
def applyFunctionX2WithConstants(x, constants):
    y = (len(x),len(constants[0]))
    np.zeros(y)
    for constantIndex in range(0, len(constants[0])):
        for i in range(0, len(x)):
            y[i][constantIndex] = (constants[constantIndex] * (x[i] ** 2)) + c_2[constantIndex]
    return y

# Custom activation function for x^2
def custom_activation_x2(x):
    return x ** 2


# Neural Network Generator
def generatePrediction(x_train, y_train, x_test):
    if globalFunctionCall == 'x2':
        get_custom_objects().update({'custom_activation': Activation(custom_activation_x2)})

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='custom_activation'))
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=2, verbose=0)
    y_pred = model.predict(x_test)
    return y_pred

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
    
def initialize_X_Array(lowerBound, upperBound, dataSize , input_constant_matrix):
    x = np.linspace(lowerBound, upperBound, num=len(dataSize))
    x_array = (len(x),len(input_constant_matrix[0]))
    for i in range(0, len(input_constant_matrix[0])):
        x_array.append(x)
    return x_array

# Main
if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = initialize_X_Array(trainLowerBound,trainUpperBound,trainDataSize, trainConstantMatrix)
    x_test = initialize_X_Array(testLowerBound,testUpperBound,testDataSize, testConstantMatrix)
    y_train = applyFunctionX2WithConstants(x_train, trainConstantMatrix)
    y_test = applyFunctionX2WithConstants(x_test, testConstantMatrix)
    # Generate the prediction provided the training information
    y_pred = generatePrediction(x_train, y_train, x_test)
    # Plot the results
    plot(x_train, y_pred, y_test)
