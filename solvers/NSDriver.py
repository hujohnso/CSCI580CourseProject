import differential_equations.PDE as pde
from solver import generatePrediction, plot, calculateError, initialize_X_Array
import autograd.numpy as np
import numpy
from time import time

#numberOfNodesInLayer = np.array([2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 32943])
numberOfNodesInLayer = np.array([3000, 32943])
activationOfLayer = np.array(['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'
                                 ,'relu', 'relu', 'relu', 'relu', 'relu', 'relu'])
ClassToUse = pde.NavierStokes


def produceConstantsFromRangeAndGrain(upperBound, lowerBound, numberOfConstants):
    return numpy.linspace(lowerBound, upperBound, numberOfConstants)


if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = numpy.linspace(1, 100, 100)
    x_test = numpy.linspace(1, 1, 1)
    timeToTrainStart = time()
    myDE = ClassToUse(x_test, x_train)
    print('The time taken to generate the arrays is: ', time() - timeToTrainStart, ' seconds')
    # Generate the prediction provided the training information

    y_pred = generatePrediction(myDE, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer, activationOfLayer)

    np.savetxt('NavierStokes.dat', y_pred)
    np.savetxt('NavierStokesTrue.dat', myDE.y_train[50])
    #Plot the results
    # plot(myDE, y_pred)
    #calculateError(y_pred,myDE)
