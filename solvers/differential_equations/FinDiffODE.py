import autograd.numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import copy

class finDiffODE1:

    def __init__ (self, x_test, x_train):
        self.x_test = x_test
        self.x_train = x_train

    def funcApply(self, x):
        t = copy.deepcopy(x)
        t = t.reshape(1, len(x))
        t = t[0]
        y = odeint(self.model, self.y0, t)
        if len(self.y0) > 1:
            y = y[:, 0]
        y.shape = len(x)
        return y

    def performFunction(self, x, constants):
        constantsLength = constants.shape[0]
        y = np.zeros(shape=(len(x),constantsLength))
        self.constants = constants
        for j in range(0, constantsLength):
            self.iteration = j
            y[:,j] = self.funcApply(x[:,j])
        return y


# Test with analytical DE, dy/dt = -k*y
# Analytical solution is y = ce^{-kt}
# INITIAL CONDITION MUST BE y0 > 0
# Performs best with relu-relu-relu-custom
class neg_ex(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y0 = [1]

    def model(self, y, t):
        k = self.constants[self.iteration][0]
        dydt = -k * y
        return dydt

    def func(self, x, constants=np.array(1)):
        return super(neg_ex, self).performFunction(x, constants)

    def custom_activation(self, x):
        return tf.math.exp(-self.constants[self.iteration][0]*x)

# Test with DE dy/dt = 3e^{-t}
class neg_ex_2(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y0 = [1]

    def model(self, y, t):
        k = self.constants[self.iteration][0]
        dydt = k * np.exp(-t)
        return dydt

    def func(self, x, constants=np.array(1)):
        return super(neg_ex_2, self).performFunction(x, constants)

    def custom_activation(self, x):
        return self.constants[self.iteration][0] - tf.nn.relu(x)

# Test with DE dy/dt = x(1+|x|)
class x2(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y0 = [1]

    def model(self, y, t):
        dydt = t*(1+(np.abs(t)))
        return dydt

    def func(self, x, constants=np.array(1)):
        return super(x2, self).performFunction(x, constants)

    def custom_activation(self, x):
        return x ** 2

# Test with DE dy/dt = x|sin(x)|
class abssin(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y0 = [1]

    def model(self, y, t):
        dydt = t*np.abs(np.sin(t))
        return dydt

    def func(self, x, constants=np.array(1)):
        return super(abssin, self).performFunction(x, constants)

    def custom_activation(self, x):
        return x**2 + tf.cos(x) * x

# y'' + y' + 2y = 0
# x1' = x2
# x2' = -2x1 - x2
# x1(0) = 1, x2(0) = 0
class secondOrder1(finDiffODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)
        self.y0 = [1, 0]

    def model(self, y, t):
        dydt = [y[1], -2*y[0]-y[1]]
        return dydt

    def func(self, x, constants=np.array(1)):
        return super(secondOrder1, self).performFunction(x, constants)

    def custom_activation(self, x):
        return tf.math.exp(-x)
