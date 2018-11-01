import autograd.numpy as np
import tensorflow as tf

class analyticalODE1:

    def __init__ (self, x_test, x_train):
        self.x_test = x_test
        self.x_train = x_train

class x2(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = x[i] ** 2 / 2
        return y

    @staticmethod
    def custom_activation(x):
        return x ** 2


class x4(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = x[i] ** 4 / 4 + x[i] ** 3 / 3 + x[i] ** 2 / 2
        return y

    @staticmethod
    def custom_activation(x):
        return x ** 4


class ex(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = np.exp(x[i])
        return y

    @staticmethod
    def custom_activation(x):
        return tf.math.exp(x)


class log(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = log(x[i])
        return y

    @staticmethod
    def custom_activation(x):
        return tf.log(x)
