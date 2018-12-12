import differential_equations.PDE as pde
from solver import generatePrediction, plot, calculateError, initialize_X_Array
import autograd.numpy as np
import numpy
from time import time

numberOfNodesInLayer = np.array([20, 8100])
numberOfNodesInLayer_More = np.array([20, 8281])
activationOfLayer = np.array(['relu','relu'])

if __name__ == "__main__":
    numpy.random.seed(7)
    x_train = numpy.linspace(1, 100, 100)
    x_test = numpy.linspace(1, 1, 1)

    timeToTrainStart = time()
    Q = pde.NavierStokesQ(x_test, x_train)
    P = pde.NavierStokesP(x_test, x_train)
    #VFFA = pde.NavierStokesVFFA(x_test, x_train)
    #VFSA = pde.NavierStokesVFSA(x_test, x_train)
    print('The time taken to generate the arrays is: ', time() - timeToTrainStart, ' seconds')
    # Generate the prediction provided the training information

    y_Q = generatePrediction(Q, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer_More, activationOfLayer)
    y_P = generatePrediction(P, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer, activationOfLayer)
    #y_VFFA = generatePrediction(VFFA, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer_More, activationOfLayer)
    #y_VFSA = generatePrediction(VFSA, np.array([[1]]), np.array([[1]]), numberOfNodesInLayer_More, activationOfLayer)


    #np.savetxt('Q.dat', y_Q)
    #np.savetxt('P.dat', y_P)
    #np.savetxt('VFFA.dat', y_VFFA)
    #np.savetxt('VFSA.dat', y_VFSA)

    #np.savetxt('Q_True.dat', Q.y_train[1])
    #np.savetxt('P_True.dat', P.y_train[1])
    #np.savetxt('VFFA_True.dat', VFFA.y_train[50])
    #np.savetxt('VFSA_True.dat', VFSA.y_train[50])
