import differential_equations.PDE as pde
from solver import generatePrediction, plot, calculateError, initialize_X_Array
import autograd.numpy as np
import numpy
import time

trainUpperBound = 10
trainLowerBound = 0
testUpperBound = 10
testLowerBound = 0
trainDataSize = 500
testDataSize = 100

numberOfNodesInLayer = np.array([100,  8100])
activationOfLayer = np.array(['relu', 'relu', 'relu'])
ClassToUse = pde.NavierStokes

def produceConstantsFromRangeAndGrain(upperBound, lowerBound, numberOfConstants):
    return numpy.linspace(lowerBound, upperBound, numberOfConstants)


if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = numpy.linspace(1, 400, 400)
    x_test = numpy.linspace(1, 1, 1)

    myDE = ClassToUse(x_test, x_train)

    # Generate the prediction provided the training information
    y_pred = generatePrediction(myDE, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer, activationOfLayer)

    np.savetxt('NSoutput.dat', y_pred)

     #Plot the results
    plot(myDE, y_pred)
    #calculateError(y_pred,myDE)
