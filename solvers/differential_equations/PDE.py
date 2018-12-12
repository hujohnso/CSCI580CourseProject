import os
import autograd.numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import copy

class PDE:

    def __init__ (self, x_test, x_train):
        self.x_test = x_test
        self.x_train = x_train

class heatEqn(PDE):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y_train = []
        for i in range(1, 1001):
            with open(os.path.join('HeatEqnData', str(i) + '.csv')) as f:
                reader = f.readlines()
                timestamp = []
                for row in reader:
                    split = row.split(',')
                    for s in split:
                        timestamp.append(float(s))
                self.y_train.append(timestamp)

        self.y_test = self.y_train[1]
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)

# Were going to try to just append the four significant values to the end of one big
# 8100*4 vector for each time step
class NavierStokes(PDE):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y_train = []
        foldersToRead = ['PressureField', 'StreamFunction', 'NormalizedVelocityFieldFirstArg', 'NormalizedVelocityFieldSecondArg']
        for i in range(1, 101):
            timeArray = []
            for fileName in foldersToRead:
                with open(os.path.join('/home/hujohnso/Documents/CSCI580/CSCI580CourseProject/solvers/differential_equations/'
                                   'NavierStokesData/' + fileName,
                                       str(i) + '.csv')) as f:
                    reader = f.readlines()
                    for row in reader:
                        split = row.split(',')
                        for s in split:
                            timeArray.append(float(s))
            self.y_train.append(timeArray)

        self.y_test = self.y_train[50]
        #self.y_test = [self.y_test]
        self.y_test = np.array(self.y_test)
        #self.y_train = [self.y_train]
        self.y_train = np.array(self.y_train)
