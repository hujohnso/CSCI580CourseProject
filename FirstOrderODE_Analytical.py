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
trainUpperBound = 90
trainLowerBound = 10
testUpperBound = 1000
testLowerBound = 100
trainDataSize = 2000
testDataSize = 100

# Test differential equation is dy/dx = 1/x
# The analytical solution to this equation is y = log(x) + c (we will ignore the contants)
def applyFunctionLog(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = log(x[i], 2)
    return y

# Test differential equation is dy/dx = x
# The analytical solution to this equation is y = x^2/2 + c (we will ignore the contants)
def applyFunctionX2(x):
    y = np.linspace(0, 1, num=len(x))
    y.shape = len(x), 1
    for i in range(0, len(x)):
        y[i] = x[i] ** 2 / 2
    return y

# Find the maximum value in the vector
def findMaximum(y):
    maxVal = [-1];
    for i in range(0, len(y)):
        if y[i] > maxVal:
            maxVal = y[i]
    return maxVal

# Adjusting the y vector to be between 0 and 1
def adjustVector(y, maximumValue):
    for i in range(0, len(y)):
        y[i] = y[i] / maximumValue
    return y

# re-Adjusting the y vector to be actual values
def reAdjustVector(y, maxVal):
    for i in range(0, len(y)):
        y[i] = y[i] * maxVal
    return y

# Neural Network Generator
def generatePrediction(x_train, y_train, x_test):
    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))

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

    y_train_temp = deepcopy(y_train)
    maxValueY = findMaximum(y_train_temp)
    y_train = adjustVector(y_train, maxValueY)

    y_test_temp = deepcopy(y_test)
    maxValueY_test = findMaximum(y_test_temp)
    y_test = adjustVector(y_test, maxValueY_test)

    # Generate the prediction provided the training information
    y_pred = generatePrediction(x_train, y_train, x_test)

    # Re-adjust output
    y_test = reAdjustVector(y_test, maxValueY_test)
    y_pred = reAdjustVector(y_pred, maxValueY_test)

    # Plot the results
    plot(x_train, y_pred, y_test)

