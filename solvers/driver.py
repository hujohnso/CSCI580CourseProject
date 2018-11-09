#!/usr/bin/env python3
import differential_equations.AnalyticalODE1 as ode
import differential_equations.FinDiffODE1 as findiffODE
from solver import generatePrediction, plot, calculateError, initialize_X_Array
import autograd.numpy as np
import numpy
import time

trainUpperBound = 10
trainLowerBound = 0
testUpperBound = 10
testLowerBound = 0
trainDataSize = 500
testDataSize = 200

#rows will be the particular constants and columns will be each iteration
#testConstants = np.array([[3,2,3],[4,3,2]])
#trainConstants = np.array([[3,2,2],[3,2,2],[4,3,3]])
#testConstants = np.array([[5,5,5]])
#trainConstants = np.array([[1,1,1], [1,1,5],[1,5,1],[5,1,1], [10,5,5], [5,10,5],[5,5,10], [5,5,5]])
# ClassToUse = ode.x4

testConstants = np.array([[4.9]])
trainConstants = np.array([[4.8], [4.9], [5]])
#ClassToUse = ode.x2
ClassToUse = findiffODE.x2

# testConstants = np.array([[3],[4],[5]])
# trainConstants = np.array([[3],[4],[5]])
# ClassToUse = ode.ex

#testConstants = np.array([[3],[4],[5]])
#testConstants = np.array([[4]])
#trainConstants = np.array([[3.97], [3.99], [4.01], [4.03]])
#ClassToUse = ode.log

if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = initialize_X_Array(trainLowerBound,trainUpperBound,trainDataSize,trainConstants)
    x_test = initialize_X_Array(testLowerBound,testUpperBound,testDataSize,testConstants)
    myODE = ClassToUse(x_test, x_train)

    myODE.y_train = ClassToUse.func(myODE, x_train, trainConstants)
    startFinDiff = time.clock()
    myODE.y_test = ClassToUse.func(myODE, x_test ,testConstants)
    print("Total time for finite difference approximation is: ", time.clock() - startFinDiff)

    #x_train.shape = trainDataSize, 1
    #x_test.shape = testDataSize, 1

    # Generate the prediction provided the training information
    y_pred = generatePrediction(myODE, trainConstants, testConstants)

    # Plot the results
    plot(myODE, y_pred)
    calculateError(y_pred,myODE)
