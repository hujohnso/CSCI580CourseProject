import autograd.numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import copy

class finDiffODE1:

    def __init__ (self, x_test, x_train):
        self.x_test = x_test
        self.x_train = x_train

# Test with analytical DE, dy/dt = -k*y
# Analytical solution is y = ce^{-kt}
class neg_ex(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def model(y, t):
        k = 0.5 # This will be an input param in future integration
        dydt = -k * y
        return dydt

    def func(x, model):
        t = copy.deepcopy(x)
        y0 = 1 # initial condition
        y = odeint(model, y0, t)
        y.shape = len(x), 1
        return y

    @staticmethod
    def custom_activation(x):
        return tf.math.exp(-x)

# Test with DE dy/dt = 3e^{-t}
class neg_ex_2(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def model(y, t):
        dydt = 3.0 * np.exp(-t)
        return dydt

    def func(x, model):
        t = copy.deepcopy(x)
        y0 = 0  # initial condition
        y = odeint(model, y0, t)
        y.shape = len(x), 1
        return y

    @staticmethod
    def custom_activation(x):
        return 3 - tf.nn.relu(x)
        return 3 - tf.math.exp(-x)
