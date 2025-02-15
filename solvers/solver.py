#!/usr/bin/env python3
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import autograd.numpy as np
import time
import tensorflow as tf


#This looks a little confusing but all it is doing is combining the x_Train_Matrix with the proper constants
#that correspond to the evaluation
def pairConstantsMatrixAndXMatrixForDNNInput(x_Matrix, constantsMatrix):
    constantsLength = constantsMatrix.shape[0]
    input_Matrix = np.zeros(shape=(constantsMatrix.shape[1] + 1, len(x_Matrix) * constantsLength))
    input_Matrix[0,:] = x_Matrix.flatten('F')
    constantsExpandedMatrix = np.zeros(shape = (constantsMatrix.shape[1],len(x_Matrix) * constantsLength))
    for i in range(constantsMatrix.shape[1]):
        for j in range(constantsLength):
            input_Matrix[i + 1,len(x_Matrix) * j:len(x_Matrix) * (j + 1)] = np.repeat(constantsMatrix[j,i],len(x_Matrix))
    return input_Matrix

def generateParameterizedModel(myODE, trainConstants, numberOfNodesInLayer, activationOfLayer):
    model = Sequential()
    for i in range(len(numberOfNodesInLayer)):
        if i == 0:
            #model.add(Dense(units=numberOfNodesInLayer[i], input_dim = trainConstants.shape[1] + 1))
            model.add(Dense(units=numberOfNodesInLayer[i], input_dim = 1))
            model.add(Activation(activationOfLayer[i]))
        elif i == len(numberOfNodesInLayer) - 1:
            model.add(Dense(units=numberOfNodesInLayer[i]))
        else:
            model.add(Dense(units=numberOfNodesInLayer[i]))
            model.add(Activation(activationOfLayer[i]))

    model.compile(loss='mean_squared_error',
              optimizer='adam')
    return model


def generatePrediction(myODE, trainConstants, testConstants,numberOfNodesInLayer, activationOfLayer):
    startNNTrain = time.clock()
    model = generateParameterizedModel(myODE,trainConstants,numberOfNodesInLayer,activationOfLayer)
    #x_input_for_model = pairConstantsMatrixAndXMatrixForDNNInput(myODE.x_train,trainConstants).transpose()
    timeForTraining = time.clock()
    model.fit(myODE.x_train, myODE.y_train, epochs=30, batch_size=1, verbose=1, shuffle=True)# validation_data=(myODE.x_train,myODE.y_train))
    print("The total time for training is: ", time.clock() - timeForTraining)
    #x_input_for_model = pairConstantsMatrixAndXMatrixForDNNInput(myODE.x_test, testConstants).transpose()
    print("Total time for NN Training is: ", time.clock() - startNNTrain)
    startNN = time.clock()
    #model.save('navierStokes_Q.h5')
    y_pred = model.predict(myODE.x_test)
    #y_pred = model.predict(x_input_for_model)
    print("Total time for NN approximation is: ", time.clock() - startNN)
    return y_pred
    #return y_pred.reshape(myODE.x_test.shape[0], myODE.x_test.shape[1],order='F')

def plotMatrixData(x_test, y_test, line, labelValue):
    for i in range(x_test.shape[1]):
        plt.plot(x_test[:,i].transpose(),y_test[:,i].transpose(), line , label = labelValue)

def plot(myODE, y_pred):
    plotMatrixData(myODE.x_test,myODE.y_test, 'r--', 'analytical soln')
    plotMatrixData(myODE.x_test,y_pred,'b','DNN soln')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("dy/dx = x|sin(x)|")
    ##plt.xlim([myODE.x_test[0], myODE.x_test[len(myODE.x_test)-1]])
    plt.savefig('image.png')

def calculateError(y_pred, myODE):
    localError = 0
    for i in range(myODE.y_test.shape[0]):
        for j in range(myODE.y_test.shape[1]):
            if myODE.y_test[i,j] == 0:
                continue
            localError += abs((y_pred[i,j] - myODE.y_test[i,j]) /myODE. y_test[i,j]) * 100
    avgError = localError / (myODE.y_test.size)
    print("On average, the error is: ", avgError, " %")

def initialize_X_Array(lowerBound, upperBound, dataSize , constants):
    constantsLength = constants.shape[0]
    x = np.linspace(lowerBound, upperBound, num= dataSize)
    x.shape = len(x),1
    x_array = np.zeros(shape=(len(x),constantsLength))
    for j in range(0, constantsLength):
        x_array[:,j] = list(x)
    return x_array

