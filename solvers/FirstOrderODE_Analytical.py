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
trainDataSize = 5
testDataSize = 100
#rows will be the particular constants and columns will be each iteration
testConstants = np.array([[3],[4]])
trainConstants = np.array([[1],[2],[3]])
constant = None
odeClassToUse = ode.x2

#This looks a little confusing but all it is doing is combining the x_Train_Matrix with the proper constants
#that correspond to the evaluation
def pairConstantsMatrixAndXMatrixForDNNInput(x_Matrix, constantsMatrix):
    constantsLength = getConstantsLength(constantsMatrix)
    input_Matrix = y = np.zeros(shape=(constantsMatrix.shape[1] + 1, len(x_Matrix) * constantsLength))
    input_Matrix[0,:] = x_Matrix.flatten('F')
    constantsExpandedMatrix = np.zeros(shape = (constantsMatrix.shape[1],len(x_Matrix) * constantsLength))
    for i in range(constantsMatrix.shape[1]):
        for j in range(constantsLength):
            input_Matrix[i + 1,len(x_Matrix) * j:len(x_Matrix) * (j + 1)] = np.repeat(constantsMatrix[j,:],len(x_Matrix))
    return input_Matrix
def generatePrediction(myODE):

    get_custom_objects().update({'custom_activation': Activation(myODE.custom_activation)})

    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim = trainConstants.shape[1] + 1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='custom_activation'))

    #model.output_shape   #model.summary()  #model.get_config()  #model.get_weights()

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    
    x_input_for_model = pairConstantsMatrixAndXMatrixForDNNInput(myODE.x_train,trainConstants).transpose()
    model.fit(x_input_for_model, myODE.y_train.flatten('C').reshape(-1,1), epochs=20, batch_size=2, verbose=0)
    x_input_for_model = pairConstantsMatrixAndXMatrixForDNNInput(myODE.x_test, testConstants).transpose()
    y_pred = model.predict(x_input_for_model)
    return y_pred.reshape(x_test.shape[0],x_test.shape[1],order='F')

def plotMatrixData(x_test, y_test, line, labelValue):
    for i in range(x_test.shape[1]):
        plt.plot(x_test[:,i].transpose(),y_test[:,i].transpose(), line , label = labelValue)
 #y_pred.reshape(x_test.shape[0],x_test.shape[1],order='F')
def plot(myODE, y_pred):
    tfit = np.linspace(testLowerBound, testUpperBound, num=testDataSize).reshape(-1, 1)
    plotMatrixData(myODE.x_test,myODE.y_test, 'r--', 'analytical soln')
    plotMatrixData(myODE.x_test,y_pred,'b--','DNN soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([testLowerBound, testUpperBound])
    plt.savefig('image.png')

def calculateError(y_pred, myODE):
    localError = 0
    for i in range(myODE.y_test.shape[0]):
        for j in range(myODE.y_test.shape[1]):
            localError += abs((y_pred[i,j] - myODE.y_test[i,j]) /myODE. y_test[i,j]) * 100
    avgError = localError / (myODE.y_test.size)
    print("On average, the error is: ", avgError, " %")
    
def initialize_X_Array(lowerBound, upperBound, dataSize , constants):
    constantsLength = getConstantsLength(constants)
    x = np.linspace(lowerBound, upperBound, num= dataSize)
    x.shape = len(x),1
    x_array = np.zeros(shape=(len(x),constantsLength))
    for j in range(0, constantsLength):
        x_array[:,j] = list(x)
    return x_array

#This is necessary to incorporate the case in which constants is set to none
def getConstantsLength(constants):
    if constants is None:
        constantsLength = 1
    else:
        constantsLength = constants.shape[0]
    return constantsLength

if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = initialize_X_Array(trainLowerBound,trainUpperBound,trainDataSize,trainConstants)
    x_test = initialize_X_Array(testLowerBound,testUpperBound,testDataSize,testConstants)
    myODE = odeClassToUse(x_test, x_train)
    myODE.y_train = odeClassToUse.func(myODE, x_train,trainConstants)
    myODE.y_test = odeClassToUse.func(myODE,x_test,testConstants)

    y_pred = generatePrediction(myODE)

    plot(myODE, y_pred)
    
    calculateError(y_pred,myODE)
    
    
