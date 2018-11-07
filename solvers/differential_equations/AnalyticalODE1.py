import autograd.numpy as np
import tensorflow as tf

class analyticalODE1:
    def __init__ (self, x_test, x_train):
        self.x_test = x_test
        self.x_train = x_train

    #I didn't feel like repeating the logic for all of the seperate functions
    #so here is a generic evaluator for all of the sub classes to use :)
    def performAnalyticalFunction(self ,functionToPerform, x , constants):
        if constants is None:
            constantsLength = 1
        else:
            constantsLength = constants.shape[0]
        y = np.zeros(shape=(len(x),constantsLength))
        for j in range(0, constantsLength):
            for i in range(0, len(x)):
                y[i,j] = functionToPerform(x[i,j],constants[j,:])
        return y

class x2(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(self,x,constants=np.array(1)):
        return super(x2, self).performAnalyticalFunction(lambda value, constant: constant * (value ** 2 / 2), x, constants)

    @staticmethod
    def custom_activation(x):
        return x ** 2

class x4(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(self,x,constants=np.array([1,1,1])):
        return super(x4, self).performAnalyticalFunction(lambda value ,constant:(constant[0] * (value ** 4 / 4)) + constant[1] * (value ** 3 / 3) + constant[2] * (value ** 2 / 2), x, constants)

    @staticmethod
    def custom_activation(x):
        return x ** 4


class ex(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(self, x,constants=np.array([1])):
        return super(ex, self).performAnalyticalFunction(lambda value, constant: constant * np.exp(value), x, constants)

    @staticmethod
    def custom_activation(x):
        return tf.math.exp(x)


class log(analyticalODE1):
    def __init__(self, x_test, x_train):
        super().__init__(x_test, x_train)

    def func(self, x,constants=np.array([1])):
        return super(log, self).performAnalyticalFunction(lambda value,constant: constant * np.log(value), x, constants)

    @staticmethod
    def custom_activation(x):
        return tf.log(x)
