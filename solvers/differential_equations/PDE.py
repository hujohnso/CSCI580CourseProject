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
        for i in range(1, 401):
            with open(os.path.join('HeatEqnData', str(i) + '.csv')) as f:
                reader = f.readlines()
                timestamp = []
                for row in reader:
                    split = row.split(',')
                    for s in split:
                        timestamp.append(float(s))
                self.y_train.append(timestamp)

        self.y_test = self.y_train[0]
        #self.y_test = [self.y_test]
        self.y_test = np.array(self.y_test)
        #self.y_train = [self.y_train]
        self.y_train = np.array(self.y_train)


class NavierStokes(PDE):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y_train = []
        for i in range(1, 401):
            with open(os.path.join('/home/hujohnso/Documents/CSCI580/CSCI580CourseProject/solvers/differential_equations/'
                                   'NavierStokesData/Pressure', str(i) + '.csv')) as f:
                reader = f.readlines()
                timestamp = []
                for row in reader:
                    split = row.split(',')
                    for s in split:
                        timestamp.append(float(s))
                self.y_train.append(timestamp)

        self.y_test = self.y_train[0]
        #self.y_test = [self.y_test]
        self.y_test = np.array(self.y_test)
        #self.y_train = [self.y_train]
        self.y_train = np.array(self.y_train)
