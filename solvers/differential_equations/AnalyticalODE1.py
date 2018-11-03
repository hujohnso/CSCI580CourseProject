import autograd.numpy as np
import tensorflow as tf

class analyticalODE1:
    constant = None
    def __init__ (self, x_test, x_train, constant = 1):
        self.x_test = x_test
        self.x_train = x_train
        self.constant = constant

class x2(analyticalODE1):
    def __init__(self, x_test, x_train,constant = 1):
        super().__init__(x_test, x_train,constant)
        
    def func(self,x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = self.constant * (x[i] ** 2 / 2)
        return y

    @staticmethod
    def custom_activation(x):
        return x ** 2

class x4(analyticalODE1):
    def __init__(self, x_test, x_train, constant = np.array([1,1,1])):
        super().__init__(x_test, x_train, constant)

    def func(self,x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = (self.constant[0] * (x[i] ** 4 / 4)) + self.constant[1] * (x[i] ** 3 / 3) + self.constant[2] * (x[i] ** 2 / 2)
        return y

    @staticmethod
    def custom_activation(x):
        return x ** 4


class ex(analyticalODE1):
    def __init__(self, x_test, x_train, constant = 1):
        super().__init__(x_test, x_train, constant)

    def func(self, x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = self.constant * np.exp(x[i])
        return y

    @staticmethod
    def custom_activation(x):
        return tf.math.exp(x)


class log(analyticalODE1):
    def __init__(self, x_test, x_train,constant = 1):
        super().__init__(x_test, x_train,constant)

    def func(self, x):
        y = np.linspace(0,0,num=len(x))
        y.shape = len(x), 1
        for i in range(0, len(x)):
            y[i] = self.constant * log(x[i])
        return y

    @staticmethod
    def custom_activation(x):
        return tf.log(x)
