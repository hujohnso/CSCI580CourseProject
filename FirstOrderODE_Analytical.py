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

globalFunctionCall = 'log'


# Test differential equation is dy/dx = 1/x
# The analytical solution to this equation is y = log(x) + c (we will ignore the contants)
def applyFunctionLog(x):
    y = np.linspace(0, 1)
    y.shape = 50, 1
    for i in range(0, 50):
        y[i] = log(x[i], 2)
    return y

# Test differential equation is dy/dx = x
# The analytical solution to this equation is y = x^2/2 + c (we will ignore the contants)
def applyFunctionX2(x):
    y = np.linspace(0, 1)
    y.shape = 50, 1
    for i in range(0, 50):
        y[i] = x[i] ** 2 / 2
    return y

# Find the maximum value in the vector
def findMaximum(y):
    maxVal = [-1];
    for i in range(0, 50):
        if y[i] > maxVal:
            maxVal = y[i]
    return maxVal

# Adjusting the y vector to be between 0 and 1
def adjustVector(y, maximumValue):
    for i in range(0, 50):
        y[i] = y[i] / maximumValue
    return y

# re-Adjusting the y vector to be actual values
def reAdjustVector(y, maxVal):
    for i in range(0, 50):
        y[i] = y[i] * maxVal
    return y

# Neural Network Generator
def generatePrediction(x_train, y_train):
    # Set up the Neural Network Model
    model = Sequential()
    # First layer takes in 1 input
    model.add(Dense(1, activation='relu', input_dim = 1))
    # Middle layer with 10 nodes
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    # Output with 1 node
    model.add(Dense(1, activation='sigmoid'))
    #model.output_shape
    #model.summary()
    #model.get_config()
    #model.get_weights()
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=30, batch_size=5, verbose=0)

    # y_pred contains the prediction with the x_test input
    y_pred = model.predict(x_train)
    score = model.evaluate(x_train, y_train,verbose=1)
    print(score)
    return y_pred

# Plotting function
def plot(x, y_pred, y_analytical):
    tfit = np.linspace(1, 20).reshape(-1, 1)
    plt.plot(tfit, y_pred, label='soln')
    plt.plot(tfit, y_analytical, 'r--', label='analytical soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 20])
    name = globalFunctionCall + '.png'
    plt.savefig(name)

# Main
if __name__ == "__main__":
    numpy.random.seed(7)

    # The test x input is linear spacing from 0 to 20
    x_train = np.linspace(1,20)
    x_train.shape = 50, 1

    if globalFunctionCall == 'x2':
        y_train = applyFunctionX2(x_train)
    elif globalFunctionCall == 'log':
        y_train = applyFunctionLog(x_train)

    y_train_temp = deepcopy(y_train)
    maxValueY = findMaximum(y_train_temp)
    y_train = adjustVector(y_train, maxValueY)

    # Generate the prediction provided the training information
    y_pred = generatePrediction(x_train, y_train)

    # Re-adjust output
    y_train = reAdjustVector(y_train, maxValueY)
    y_pred = reAdjustVector(y_pred, maxValueY)

    # Plot the results
    plot(x_train, y_pred, y_train)

