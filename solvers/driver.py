#!/usr/bin/env python3
import differential_equations.AnalyticalODE1 as ode
import differential_equations.FinDiffODE1 as findiffODE
from solver import generatePrediction, plot
import autograd.numpy as np
import numpy
import time

trainUpperBound = 10
trainLowerBound = 0
testUpperBound = 15
testLowerBound = 0
trainDataSize = 300
testDataSize = 100
DE_Selection = findiffODE.neg_ex_2

if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = np.linspace(trainLowerBound, trainUpperBound, num=trainDataSize)
    x_test = np.linspace(testLowerBound, testUpperBound, num=testDataSize)

    # Fin Diff ODE need a second argument
    myODE = DE_Selection(x_test, x_train)
    myODE.y_test = DE_Selection.func(x_test, DE_Selection.model)

    startFinDiff = time.clock()
    myODE.y_train = DE_Selection.func(x_train, DE_Selection.model)
    print("Total time for finite difference approximation is: ", time.clock() - startFinDiff)

    x_train.shape = trainDataSize, 1
    x_test.shape = testDataSize, 1

    # Generate the prediction provided the training information
    y_pred = generatePrediction(myODE)

    # Plot the results
    plot(myODE, y_pred)
